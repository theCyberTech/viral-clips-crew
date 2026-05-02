# Standard library imports
from pathlib import Path
import os
import warnings
import logging

# Third party imports
import torch
import whisper
from whisper.utils import get_writer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")


def transcribe_file(model, srt, plain, file, output_dir="whisper_output"):
    input_file_path = Path(file)
    logging.info(f"Transcribing file: {input_file_path}\n")

    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run Whisper
    result = model.transcribe(str(input_file_path), fp16=False, verbose=False, language="en")

    output_file_name = input_file_path.stem

    if plain:
        txt_path = output_dir / f"{output_file_name}.txt"
        logging.info(f"Creating text file: {txt_path}")

        with open(txt_path, "w", encoding="utf-8") as txt:
            txt.write(result["text"])

        transcript = result["text"]

    if srt:
        logging.info("Creating SRT file")
        srt_writer = get_writer("srt", str(output_dir))
        srt_writer(result, output_file_name)

        # Construct the SRT file path manually
        srt_path = output_dir / f"{output_file_name}.srt"

        # Read the SRT subtitles from the generated file
        with open(srt_path, "r", encoding="utf-8") as srt_file:
            subtitles = srt_file.read()

    return result, transcript, subtitles


def transcribe_main(file, output_dir="whisper_output"):

    # specify the type of file outputs you need from Whisper
    plain = True
    srt = True

    # Whisper configuration

    # Use CUDA, if available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the desired model
    model = whisper.load_model("medium.en").to(DEVICE)

    result, transcript, subtitles = transcribe_file(model, srt, plain, file, output_dir)

    return transcript, subtitles


def local_whisper_process(input_folder, output_folder):
    """Transcribe all .mp4 files in input_folder using local Whisper.

    Args:
        input_folder: Directory containing .mp4 files to transcribe.
        output_folder: Directory to write .srt and .txt output files.
    """
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_video_path = os.path.join(input_folder, filename)
            logging.info(f"Processing video: {input_video_path}")

            transcribe_main(input_video_path, output_dir=output_folder)

    logging.info("local_transcribe.py completed")


if __name__ == "__main__":
    input_folder = 'input_files'
    output_folder = 'whisper_output'

    if os.path.exists(input_folder):
        local_whisper_process(input_folder, output_folder)
    else:
        logging.error(f"Input folder not found: {input_folder}")
