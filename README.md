<a href="https://x.com/alxfazio" target="_blank">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/vcc-github-banner.png">
    <img alt="Viral Clips Crew Banner" src="images/vcc-github-banner.png" width="400px" style="max-width: 100%; margin-bottom: 20px;">
  </picture>
</a>

# Viral Clips Crew

A [CrewAI](https://github.com/joaomdmoura/crewAI)-powered video editing assistant that watches long-form content and extracts the most engaging, potentially viral segments — trimmed, subtitled, and ready for social media.

> [!note]
> Originally created by [Alex Fazio](https://x.com/alxfazio) as a weekend hack. This fork extends and refines the pipeline with local transcription, improved clipping logic, and a cleanup utility.

## How It Works

The pipeline has six stages:

```
Input → Transcribe → Extract → Match → Clip → Subtitle
```

1. **Input** — Choose a YouTube URL (auto-downloaded via `yt-dlp`) or an existing `.mp4` file in `input_files/`.
2. **Transcribe** — YouTube videos use the transcript API when available; local files use [OpenAI Whisper](https://github.com/openai/whisper) (`medium.en` model). Outputs `.srt` and `.txt` files to `whisper_output/`.
3. **Extract** — GPT-4o reads the full transcript and identifies the top 3 clips with the highest viral potential, returning structured JSON.
4. **Match** — A CrewAI agent powered by Gemini (`gemini-1.5-pro-exp-0801`) matches each extract to its exact `.srt` subtitle timestamps.
5. **Clip** — `ffmpeg` trims the video into 30–150 second segments, optionally cropping to 1:1 (square) for social platforms. Outputs to `clipper_output/`.
6. **Subtitle** — Subtitles are burned into the trimmed clips using `ffmpeg` with adjusted timings. Final output lands in `subtitler_output/`.

## Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Main orchestrator — runs the full pipeline end-to-end |
| `ytdl.py` | Downloads YouTube videos and fetches transcripts via YouTube Transcript API |
| `local_transcribe.py` | Transcribes local `.mp4` files using OpenAI Whisper locally |
| `extracts.py` | Calls OpenAI (GPT-4o) to identify the 3 most viral-worthy segments from the transcript |
| `crew.py` | CrewAI agent (Gemini) matches extract text to `.srt` timing segments |
| `clipper.py` | Trims video to matched segments using `ffmpeg`; supports 1:1 square cropping |
| `subtitler.py` | Adjusts subtitle timings and burns them into the final video |
| `reboot.py` | Cleanup utility — moves intermediate files to trash for a fresh run |
| `utils.py` | File-lock helper for multi-process coordination |
| `resources/` | Prompt templates and YouTube URL format reference |

## Requirements

- **Python** ≥ 3.10, ≤ 3.13 (managed with Poetry)
- **API keys**: [OpenAI](https://platform.openai.com/api-keys) and [Google Gemini](https://aistudio.google.com/apikey)
- **ffmpeg** installed and available on `$PATH`
- **CUDA-capable GPU** recommended for local Whisper (falls back to CPU)

## Installation

1. **Clone the repository:**

   ```shell
   git clone https://github.com/alexfazio/viral-clips-crew.git
   cd viral-clips-crew
   ```

2. **Install Poetry** (if not already installed):

   ```shell
   pip install poetry
   ```

3. **Install dependencies:**

   ```shell
   poetry install
   poetry update pydantic
   ```

4. **Configure API keys:**

   Copy `.env.example` to `.env` and fill in your keys:

   ```shell
   cp .env.example .env
   ```

   Then edit `.env`:

   ```env
   OPENAI_API_KEY=sk-...
   GEMINI_API_KEY=AIza...
   ```

5. **Install ffmpeg** (if not already installed):

   ```shell
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt install ffmpeg

   # Windows (via Chocolatey)
   choco install ffmpeg
   ```

## Usage

Run the full pipeline:

```shell
poetry run python app.py
```

You'll be prompted to:
1. **Choose input source** — YouTube URL or existing local video
2. **Select aspect ratio** — keep original or crop to 1:1 square

The pipeline then runs automatically. Output files appear in:

| Directory | Contents |
|-----------|----------|
| `subtitler_output/` | **Final** trimmed and subtitled `.mp4` clips |
| `clipper_output/` | Trimmed clips (pre-subtitle) |
| `crew_output/` | Matched `.srt` subtitle segments + API response JSON |
| `whisper_output/` | Raw transcription (`.srt` + `.txt`) |

### Cleanup Between Runs

To clear intermediate files and start fresh:

```shell
poetry run python reboot.py
```

This moves output files to the system trash, preserving your `input_files/PLACE_CLIPS_HERE` placeholder.

## Troubleshooting

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| `TypeError: 'NoneType' object is not iterable` | Missing or invalid API key | Check `.env` and API credit balance |
| `EnvironmentError: Required environment variable not set` | Empty `.env` values | Ensure keys are set to real values, not `"none"` or `""` |
| `No module named 'torch'` | Dependencies not installed | Run `poetry install` |
| `ffmpeg: command not found` | ffmpeg missing | Install ffmpeg (see Installation) |
| `No transcript found` for YouTube | Video has no captions | The video will still download; transcription falls back to local Whisper |
| Subtitle matching is inaccurate | Gemini couldn't align extract with SRT | Try a video with clearer speech or shorter length |

## How Individual Scripts Work

### Standalone Execution

Each module can run independently for testing or partial workflows:

```shell
# Download a YouTube video + transcript only
poetry run python ytdl.py

# Transcribe local videos only
poetry run python local_transcribe.py

# Run extract → match → clip → subtitle for pre-existing files
poetry run python crew.py

# Clip a video using an existing SRT file
poetry run python clipper.py

# Burn subtitles into a trimmed video
poetry run python subtitler.py
```

### Extract Logic (`extracts.py`)

GPT-4o receives the full transcript and is prompted to find three ~1-minute segments (≈125 words, ≈10 sentences) ranked by viral potential. The structured JSON schema enforces exactly 3 clips with text, rank, and word count. If fewer than 3 are returned, filler entries are appended.

### Subtitle Matching (`crew.py`)

Three parallel CrewAI tasks each take one extract and scan the full `.srt` file to find the best-matching subtitle segment by word/phrase overlap. Each returns properly formatted `.srt` with segment numbers, timestamps, and matched text. The agent runs with `max_iter=1` and temperature `0.0` for deterministic output.

### Clipping Logic (`clipper.py`)

Subtitle blocks within a 2-second gap are merged into single clips. Clips shorter than 30 seconds or longer than 2m30s are skipped. Videos can be cropped to 1:1 (square) centered on the original frame, or kept at their native aspect ratio.

### Subtitle Burn-in (`subtitler.py`)

Subtitle timestamps are offset so the first spoken word starts at `00:00:00,000`. The file is re-encoded to UTF-8 to avoid character issues, then burned into the video using `ffmpeg`'s `subtitles` filter. Temporary files are cleaned up automatically.

## Credits

- Original concept and code by [Alex Fazio](https://x.com/alxfazio)
- Improvements and assistance by [Rip&Tear](https://x.com/Cyb3rCh1ck3n)

## License

[MIT](https://opensource.org/licenses/MIT) — Copyright (c) 2024-present, Alex Fazio

---

[![Watch the video](https://i.imgur.com/TBD2bvj.png)](https://x.com/alxfazio/status/1791863931931078719)
