import logging
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_video_id(yt_vid_url):
    """Extract and return the YouTube video ID from a URL."""
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, yt_vid_url)
    return match.group(1) if match else None


def yt_vid_url_to_mp4(yt_vid_url, mp4_dir_save_path):
    """Download the YouTube video and save it as an MP4 file."""
    try:
        os.makedirs(mp4_dir_save_path, exist_ok=True)
        yt = YouTube(yt_vid_url)
        filename = re.sub(r'[<>:"/\\|?*]', '', yt.title.replace(" ", "_"))
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        video_path = stream.download(output_path=mp4_dir_save_path, filename=filename)
        video_file = Path(video_path)
        if video_file.suffix != '.mp4':
            video_file.rename(video_file.with_suffix('.mp4'))
        logging.info(f"Video downloaded and saved to {video_file}")
    except Exception as e:
        logging.error(f"Error downloading video: {e}")


def yt_vid_id_to_srt(transcript, srt_save_path):
    """Convert transcript to SRT format and save it."""
    try:
        srt_content = []
        for i, entry in enumerate(transcript):
            start = entry['start']
            duration = entry['duration']
            text = entry['text']

            start_time = convert_seconds_to_srt_time(start)
            end_time = convert_seconds_to_srt_time(start + duration)

            srt_content.append(f"{i + 1}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append('')

        os.makedirs(srt_save_path, exist_ok=True)
        srt_file_path = os.path.join(srt_save_path, 'subtitles.srt')
        with open(srt_file_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(srt_content))
        logging.info(f"SRT file saved to {srt_file_path}")
    except Exception as e:
        logging.error(f"Error saving SRT file: {e}")


def yt_vid_id_to_txt(transcript, txt_save_path):
    """Save the transcript to a TXT file."""
    try:
        os.makedirs(txt_save_path, exist_ok=True)
        txt_file_path = os.path.join(txt_save_path, 'transcript.txt')
        full_transcript = ' '.join(entry['text'] for entry in transcript)
        with open(txt_file_path, 'w', encoding='utf-8') as file:
            file.write(full_transcript)
        logging.info(f"Transcript saved to {txt_file_path}")
    except Exception as e:
        logging.error(f"Error saving transcript: {e}")


def convert_seconds_to_srt_time(seconds):
    """Convert seconds to SRT time format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"


def main(yt_vid_url, mp4_dir_save_path, srt_dir_save_path, txt_dir_save_path):
    yt_video_id = extract_video_id(yt_vid_url)
    if not yt_video_id:
        logging.error("Invalid YouTube URL")
        return

    try:
        transcript = YouTubeTranscriptApi.get_transcript(yt_video_id)
    except Exception as e:
        logging.error(f"Error fetching transcript: {e}")
        return

    yt_vid_url_to_mp4(yt_vid_url, mp4_dir_save_path)
    yt_vid_id_to_srt(transcript, srt_dir_save_path)
    yt_vid_id_to_txt(transcript, txt_dir_save_path)


if __name__ == "__main__":
    yt_vid_url = input("Enter the YouTube URL: ")
    mp4_dir_save_path = "./input_files"
    srt_dir_save_path = "./whisper_output"
    txt_dir_save_path = "./whisper_output"
    main(yt_vid_url, mp4_dir_save_path, srt_dir_save_path, txt_dir_save_path)