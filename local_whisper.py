import os
import warnings
import logging

# Local application imports
import extracts
import subtitler
import transcribe
import crew
from utils import wait_for_file
from clip_sub import clip_and_sub

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ignore warnings
warnings.filterwarnings("ignore")


def process_video_file(input_video_path, output_video_folder, crew_output_folder, transcribe_flag, transcript=None,
                       subtitles=None):
    """This function processes a video file."""
    logging.info(f"Processing video: {input_video_path}")
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    initial_srt_path = os.path.join(crew_output_folder, f"{video_name}_subtitles.srt")

    if transcribe_flag:
        if transcript and subtitles:
            save_subtitles(initial_srt_path, subtitles)
        else:
            transcript, subtitles = transcribe.main(input_video_path)
            save_subtitles(initial_srt_path, subtitles)
    else:
        initial_srt_path = os.path.join(crew_output_folder, f"{video_name}.srt")

    if wait_for_file(initial_srt_path):
        process_with_extracts_and_crew(input_video_path, initial_srt_path, output_video_folder, crew_output_folder)
    else:
        logging.error(f"Failed to verify the readiness of subtitles file: {initial_srt_path}")


def save_subtitles(srt_path, subtitles):
    """Save subtitles to a file."""
    try:
        with open(srt_path, 'w') as srt_file:
            srt_file.write(subtitles)
        logging.info(f"Subtitles saved to {srt_path}")
    except Exception as e:
        logging.error(f"Error saving subtitles: {e}")


def process_with_extracts_and_crew(input_video_path, initial_srt_path, output_video_folder, crew_output_folder):
    """Process video with extracts and crew."""
    try:
        extracts_response = extracts.main()
        logging.info("Extracts processed.")

        whisper_output_dir = 'whisper_output'
        srt_files = [f for f in os.listdir(whisper_output_dir) if f.endswith('.srt')]
        txt_files = [f for f in os.listdir(whisper_output_dir) if f.endswith('.txt')]

        if srt_files and txt_files:
            transcript, subtitles = read_transcript_and_subtitles(whisper_output_dir, srt_files[0], txt_files[0])
            crew.main(extracts_response, subtitles)
            logging.info("Processed with crew.")
            process_clips(input_video_path, output_video_folder, crew_output_folder)
        else:
            logging.error("No .srt or .txt files found in the whisper_output directory.")
    except Exception as e:
        logging.error(f"Error processing with extracts and crew: {e}")


def read_transcript_and_subtitles(directory, srt_file, txt_file):
    """Read transcript and subtitles from files."""
    try:
        with open(os.path.join(directory, txt_file), 'r') as file:
            transcript = file.read()
        with open(os.path.join(directory, srt_file), 'r') as file:
            subtitles = file.read()
        logging.info("Transcript and subtitles read successfully.")
        return transcript, subtitles
    except Exception as e:
        logging.error(f"Error reading transcript and subtitles: {e}")
        return None, None


def process_clips(input_video_path, output_video_folder, crew_output_folder):
    """Process video clips with subtitles."""
    for srt_filename in sorted(os.listdir(crew_output_folder)):
        if srt_filename.startswith("new_file_return_subtitles") and srt_filename.endswith(".srt"):
            subtitle_file_path = os.path.join(crew_output_folder, srt_filename)
            clip_and_sub(input_video_path, subtitle_file_path, output_video_folder)
            logging.info(f"Clip processed and saved with subtitles: {subtitle_file_path}")


def local_whisper_process(input_folder, output_video_folder, crew_output_folder, transcript=None, subtitles=None,
                          transcribe_flag=True):
    """This function processes each video file in the input folder."""
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_video_path = os.path.join(input_folder, filename)
            process_video_file(input_video_path, output_video_folder, crew_output_folder, transcribe_flag, transcript,
                               subtitles)