"""
Viral Clips Extraction Crew
----------------------------
A 4-agent CrewAI pipeline that replaces the old single-call extracts.py.

Agent roles:
  Scout    — Transcript Analyst: tags viral-candidate moments
  Editor   — Segment Extractor:  expands each candidate to ~1 min of context
  Curator  — Viral Potential Ranker: scores/ranks by platform-specific criteria
  Producer — Output Formatter:    assembles final JSON + captions/hashtags

Each agent has a narrow, verifiable job. Structured output (Pydantic models)
flows between tasks via CrewAI's `context` parameter.
"""

# Standard library imports
import os
import logging
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Dict

# Third party imports
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

# ── OpenRouter configuration ─────────────────────────────────────────────
# CrewAI uses LiteLLM under the hood, which has native OpenRouter support.
# Set OPENROUTER_API_KEY in .env or pass it via OPENAI_API_KEY + base URL.

_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
if _OPENROUTER_KEY:
    os.environ.setdefault("OPENROUTER_API_KEY", _OPENROUTER_KEY)
else:
    # Fallback: use OPENAI_API_KEY with OpenRouter base URL
    _openai_key = os.getenv("OPENAI_API_KEY")
    if _openai_key and not _OPENROUTER_KEY:
        os.environ.setdefault("OPENAI_API_KEY", _openai_key)
        os.environ.setdefault(
            "OPENAI_API_BASE", "https://openrouter.ai/api/v1"
        )

# Model references — prefixed for OpenRouter routing.
# Remove the openrouter/ prefix if using direct API keys instead.
MODEL_CHEAP = "openrouter/deepseek/deepseek-v4-pro"
MODEL_SMART = "openrouter/deepseek/deepseek-v4-pro"
MODEL_GEMINI = "openrouter/google/gemini-3-flash-preview"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")


# ── Pydantic schemas for structured data flow between agents ──────────────


class ViralCandidate(BaseModel):
    """A single moment flagged by the Scout as having viral potential."""

    excerpt: str = Field(
        description="The exact text excerpt from the transcript (2-4 sentences)"
    )
    reason: str = Field(
        description="Why this moment could go viral (emotional punch, surprise, "
        "quotability, controversy, insight density, etc.)"
    )
    viral_category: str = Field(
        description="Category: 'emotional', 'surprising', 'controversial', "
        "'insightful', 'funny', 'inspirational', or 'actionable'"
    )
    timestamp_hint: str = Field(
        description="Approximate position in the transcript (e.g. 'early', "
        "'middle', 'late', or a rough % like '30%')"
    )


class ScoutOutput(BaseModel):
    """Output from the Scout agent."""

    candidates: List[ViralCandidate] = Field(
        description="5-8 viral-candidate moments found in the transcript"
    )
    transcript_summary: str = Field(
        description="One-paragraph summary of the full transcript for downstream agents"
    )
    full_transcript: str = Field(
        description="The complete original transcript text (passed through for Editor context)"
    )


class EditedSegment(BaseModel):
    """A single segment expanded and verified by the Editor."""

    candidate_index: int = Field(
        description="Index of the original candidate from the Scout's list (0-based)"
    )
    full_text: str = Field(
        description="The ~1 minute expanded segment (~125 words, ~10 sentences). "
        "Must include the candidate excerpt surrounded by context."
    )
    word_count: int = Field(description="Actual word count of full_text")
    coherence_score: int = Field(
        description="1-10 rating of how coherent and self-contained this segment is. "
        "10 = perfect standalone clip, 1 = needs surrounding context to make sense."
    )
    hook_sentence: str = Field(
        description="The single best sentence from this segment to use as a "
        "social media hook/opening line"
    )


class EditorOutput(BaseModel):
    """Output from the Editor agent."""

    segments: List[EditedSegment] = Field(
        description="Expanded segments (one per viable candidate, 3-5 total)"
    )


class PlatformScores(BaseModel):
    """Per-platform viral potential scores."""

    tiktok: int = Field(description="0-100 score for TikTok (vertical, fast-paced)")
    youtube_shorts: int = Field(
        description="0-100 score for YouTube Shorts (60s max, broader audience)"
    )
    instagram_reels: int = Field(
        description="0-100 score for Instagram Reels (visual, lifestyle-oriented)"
    )
    linkedin: int = Field(
        description="0-100 score for LinkedIn (professional/educational)"
    )


class CuratedClip(BaseModel):
    """A single clip scored and ranked by the Curator."""

    rank: int = Field(description="Final rank (1 = highest viral potential)")
    text: str = Field(description="The segment text")
    word_count: int = Field(description="Word count of the segment")
    viral_score: int = Field(description="0-100 overall viral potential score")
    platform_scores: PlatformScores = Field(
        description="Per-platform viral potential breakdown"
    )
    best_platform: str = Field(
        description="The platform where this clip would perform best"
    )
    curator_rationale: str = Field(
        description="1-2 sentence explanation of the ranking decision"
    )


class CuratorOutput(BaseModel):
    """Output from the Curator agent."""

    clips: List[CuratedClip] = Field(
        description="Ranked clips (3-4 total, from most to least viral)"
    )


class ProducerClip(BaseModel):
    """Final formatted clip from the Producer."""

    rank: int
    text: str
    word_count: int
    viral_score: int
    best_platform: str
    suggested_caption: str = Field(
        description="Social media caption optimized for the best platform"
    )
    suggested_hashtags: List[str] = Field(
        description="5-8 relevant hashtags for the clip"
    )
    hook: str = Field(description="Opening hook sentence for the social post")


class ProducerOutput(BaseModel):
    """Final output from the Producer agent."""

    clips: List[ProducerClip] = Field(description="Final ranked clips ready for production")
    batch_caption: str = Field(
        description="A single caption that could introduce all clips as a thread/carousel"
    )


# ── Agents ────────────────────────────────────────────────────────────────


def _build_scout() -> Agent:
    return Agent(
        role="Transcript Analyst",
        goal=dedent("""
            Read the full video transcript and identify every moment with high
            viral potential. Tag each candidate with the specific reason it
            could perform well on social media (emotional punch, surprise,
            controversy, insight density, quotability, humor, or actionable
            advice). Be thorough — flag 5-8 candidates, not just the obvious
            ones. Also provide a short transcript summary to give downstream
            agents context.
        """).strip(),
        backstory=dedent("""
            You are a veteran social media scout who has analyzed thousands of
            viral videos across TikTok, YouTube Shorts, Instagram Reels, and
            LinkedIn. You have an instinct for spotting moments that stop the
            scroll — the emotional crescendo, the surprising stat, the
            contrarian take, the laugh-out-loud beat. You know that the best
            clips often come from moments others would skim past. You favor
            answers and speculations over wandering questions, concrete
            statements over vague musings.
        """).strip(),
        llm=MODEL_CHEAP,
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )


def _build_editor() -> Agent:
    return Agent(
        role="Segment Extractor",
        goal=dedent("""
            Take each viral candidate identified by the Scout and expand it
            into a ~1 minute standalone segment (~125 words, ~10 sentences).
            Include enough surrounding context that the clip makes sense on its
            own without the viewer needing the full video. Check each segment
            for coherence — is it a complete thought or does it trail off?
            Score coherence 1-10 and identify the best hook sentence for social
            media.
        """).strip(),
        backstory=dedent("""
            You are a professional video editor who specializes in repackaging
            long-form content into social media clips. You understand pacing,
            narrative arcs, and the difference between a clip that works and
            one that confuses. You've cut clips for top creators and know that
            context is king — a clip without setup falls flat, but too much
            context kills momentum. You find the sweet spot every time.
        """).strip(),
        llm=MODEL_CHEAP,
        verbose=True,
        allow_delegation=False,
        max_iter=10,
    )


def _build_curator() -> Agent:
    return Agent(
        role="Viral Potential Ranker",
        goal=dedent("""
            Review the edited segments and rank them by viral potential.
            Score each clip 0-100 overall AND provide per-platform breakdowns
            (TikTok, YouTube Shorts, Instagram Reels, LinkedIn). Determine
            which platform each clip is best suited for. Provide a brief
            rationale for every ranking decision. Only keep the top 3-4 clips
            — quality over quantity.
        """).strip(),
        backstory=dedent("""
            You are a social media strategist who has managed content calendars
            for brands with millions of followers. You understand platform
            algorithms deeply: TikTok rewards watch time and replays,
            YouTube Shorts favors clear value props in the first 3 seconds,
            Instagram Reels prioritizes visual appeal and lifestyle alignment,
            and LinkedIn values professional insight and data-backed claims.
            You don't just rank by gut feel — you score systematically and can
            defend every placement.
        """).strip(),
        llm=MODEL_SMART,
        verbose=True,
        allow_delegation=False,
        max_iter=12,
    )


def _build_producer() -> Agent:
    return Agent(
        role="Output Formatter",
        goal=dedent("""
            Take the Curator's ranked clips and produce the final structured
            output. For each clip, write an optimized social media caption and
            5-8 relevant hashtags targeting the best platform. Also write a
            batch caption that could introduce all clips as a thread or
            carousel post.
        """).strip(),
        backstory=dedent("""
            You are a social media copywriter who has written captions for
            viral posts across every major platform. You know that a great
            caption doesn't just describe the video — it adds value, asks a
            question, or sparks debate. Your hashtag strategy is surgical:
            broad enough to reach new audiences, specific enough to hit the
            right niche. You write hooks that make people stop scrolling.
        """).strip(),
        llm=MODEL_CHEAP,
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )


# ── Tasks ─────────────────────────────────────────────────────────────────


def _task_scout(agent: Agent, transcript: str) -> Task:
    return Task(
        description=dedent(f"""
            You are given the complete transcript of a video. Your job is to
            find the moments with the highest viral potential.

            TRANSCRIPT:
            <transcript>
            {transcript}
            </transcript>

            Steps:
            1. Read the entire transcript carefully.
            2. Identify 5-8 distinct moments that could go viral. Look for:
               - Emotional peaks (anger, joy, sadness, awe)
               - Surprising facts, stats, or revelations
               - Controversial or contrarian opinions
               - Dense clusters of insights
               - Highly quotable one-liners
               - Funny or relatable moments
               - Actionable advice that solves a common problem
               Prioritize ANSWERS and SPECULATIONS over open-ended questions.
            3. For each candidate, extract the exact 2-4 sentence excerpt.
            4. Categorize each (emotional/surprising/controversial/insightful/
               funny/inspirational/actionable).
            5. Note the approximate position (early/middle/late or rough %).
            6. Write a one-paragraph summary of the full transcript.

            Return your findings in the structured format.
        """).strip(),
        expected_output="A list of 5-8 viral candidates with reasons, categories, and a transcript summary.",
        agent=agent,
        output_pydantic=ScoutOutput,
    )


def _task_editor(agent: Agent, transcript: str) -> Task:
    return Task(
        description=dedent(f"""
            You are given the Scout's viral candidates plus a transcript
            summary. For each candidate, expand it into a ~1 minute standalone
            segment.

            Steps:
            1. For each candidate from the Scout:
               a. Take the excerpt and expand it with surrounding context from
                  the transcript. The final segment should be ~125 words
                  (~10 spoken sentences).
               b. Make sure the segment is self-contained — a viewer should
                  understand it without watching the full video.
               c. Rate coherence 1-10 (10 = perfect standalone clip).
               d. Identify the single best hook sentence from the segment.
            2. Keep only candidates that can form a coherent clip. Drop any
               that are too fragmented to make sense on their own.
            3. Return 3-5 edited segments.

            Important: The full original transcript is provided below.
            Use it to pull surrounding context for each candidate.
            Do not fabricate text — stay faithful to the original transcript.

            FULL ORIGINAL TRANSCRIPT:
            <transcript>
            {transcript}
            </transcript>
        """).strip(),
        expected_output="3-5 expanded, coherent segments with hook sentences.",
        agent=agent,
        output_pydantic=EditorOutput,
    )


def _task_curator(agent: Agent) -> Task:
    return Task(
        description=dedent("""
            You are given edited segments from the Editor. Your job is to score
            and rank them by viral potential.

            Steps:
            1. Review each segment carefully.
            2. Score each clip 0-100 overall.
            3. Provide per-platform scores (0-100) for:
               - TikTok: rewards fast pacing, emotional hooks, trends
               - YouTube Shorts: rewards clear value in first 3 seconds,
                 broader demographics
               - Instagram Reels: rewards visual appeal, lifestyle content,
                 aspirational messaging
               - LinkedIn: rewards professional insight, data, thought
                 leadership
            4. Determine the single best platform for each clip.
            5. Rank clips from highest to lowest viral potential.
            6. Write a 1-2 sentence rationale for each ranking.
            7. Keep only the top 3-4 clips.

            Be systematic. Consider:
            - Universal appeal vs. niche interest
            - Shareability (would someone send this to a friend?)
            - Timeliness (is the topic currently trending?)
            - Emotional impact vs. intellectual impact
        """).strip(),
        expected_output="3-4 ranked clips with scores, platform breakdowns, and rationales.",
        agent=agent,
        output_pydantic=CuratorOutput,
    )


def _task_producer(agent: Agent) -> Task:
    return Task(
        description=dedent("""
            You are given the Curator's ranked clips. Produce the final
            production-ready output.

            Steps:
            1. For each clip, write an optimized social media caption. The
               caption should:
               - Add value beyond the video content (don't just repeat it)
               - Include a hook or question to drive engagement
               - Be appropriate for the clip's best platform
               - Be 1-3 sentences (short and punchy)
            2. Write 5-8 relevant hashtags per clip. Mix broad reach hashtags
               with niche-specific ones.
            3. Write a single "batch caption" that could introduce all clips
               as a thread or carousel post (e.g., "I watched [NAME]'s full
               interview and found 4 moments that changed how I think about
               [TOPIC] 🧵").
            4. Use the clip's own hook sentence where appropriate.

            Return the final structured output.
        """).strip(),
        expected_output="Final clips with captions, hashtags, and a batch caption.",
        agent=agent,
        output_pydantic=ProducerOutput,
    )


# ── Public API ────────────────────────────────────────────────────────────


def build_extraction_crew(transcript: str) -> Crew:
    """Build and return a CrewAI crew for the extraction pipeline.

    The crew runs sequentially:
      Scout → Editor → Curator → Producer

    Each agent's structured output is passed as context to the next agent.
    """
    scout = _build_scout()
    editor = _build_editor()
    curator = _build_curator()
    producer = _build_producer()

    task_scout = _task_scout(scout, transcript)
    task_editor = _task_editor(editor, transcript)
    task_curator = _task_curator(curator)
    task_producer = _task_producer(producer)

    # Wire context: each task sees the output of all previous tasks
    task_editor.context = [task_scout]
    task_curator.context = [task_scout, task_editor]
    task_producer.context = [task_scout, task_editor, task_curator]

    crew = Crew(
        agents=[scout, editor, curator, producer],
        tasks=[task_scout, task_editor, task_curator, task_producer],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def run_extraction(transcript: str) -> Optional[ProducerOutput]:
    """Run the full extraction pipeline on a transcript.

    Args:
        transcript: The full video transcript text.

    Returns:
        ProducerOutput if successful, None otherwise.
    """
    logging.info("=== STARTING EXTRACTION CREW ===")
    crew = build_extraction_crew(transcript)

    try:
        result = crew.kickoff()
        logging.info("=== EXTRACTION CREW COMPLETE ===")

        # crew.kickoff() returns a CrewOutput. The final task's pydantic
        # output is accessible via result.pydantic
        if hasattr(result, "pydantic") and result.pydantic is not None:
            return result.pydantic
        elif hasattr(result, "tasks_output") and result.tasks_output:
            # Fallback: grab the last task's pydantic output
            last_output = result.tasks_output[-1]
            if hasattr(last_output, "pydantic") and last_output.pydantic is not None:
                return last_output.pydantic

        logging.warning("No pydantic output found in crew result. Raw: %s", result)
        return None
    except Exception as e:
        logging.error("Extraction crew failed: %s", e, exc_info=True)
        return None


def get_whisper_transcript() -> Optional[str]:
    """Read the transcript from whisper_output directory."""
    whisper_dir = Path("whisper_output")
    if not whisper_dir.exists():
        logging.error("whisper_output directory not found")
        return None

    txt_files = list(whisper_dir.glob("*.txt"))
    if not txt_files:
        logging.warning("No .txt files found in whisper_output")
        return None

    with open(txt_files[0], "r", encoding="utf-8") as f:
        return f.read()


def main():
    """Entry point — replaces old extracts.main()."""
    logging.info("Starting extraction crew pipeline...")

    transcript = get_whisper_transcript()
    if transcript is None:
        logging.error("No transcript found. Run transcription first.")
        return None

    result = run_extraction(transcript)
    if result is None:
        logging.error("Extraction failed.")
        return None

    # Save the full ProducerOutput to crew_output for reference
    output_dir = Path("crew_output")
    output_dir.mkdir(exist_ok=True)

    import json

    output_path = output_dir / "api_response.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)
    logging.info("Full extraction output saved to %s", output_path)

    # Return extracts as list of text strings (backward-compatible with
    # the old extracts.main() interface that app.py expects)
    text_extracts = [clip.text for clip in result.clips]
    logging.info("Extraction complete: %d clips produced", len(text_extracts))
    return text_extracts


if __name__ == "__main__":
    main()
