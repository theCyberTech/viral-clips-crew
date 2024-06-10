import os
import warnings
import logging
from typing import Optional

# Third-party imports
import lockfile
from dotenv import load_dotenv

# Local application imports
import clipper
import extracts
import subtitler
import transcribe
from ytdl import main as ytdl_main
from local_whisper import local_whisper_process

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ignore warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Define folder paths
FOLDERS = {
    'input': './input_files',
    'output_video': './clipper_output',
    'crew_output': './crew_output',
    'whisper_output': './whisper_output'
}


def check_required_vars(required_vars) -> None:
    """Check if the required environment variables are set. Raise an error if any are missing."""
    missing_vars = [var for var in required_vars if os.getenv(var) is None]
    if missing_vars:
        raise EnvironmentError(f"Required environment variables not set: {', '.join(missing_vars)}")


def get_user_choice() -> str:
    """Prompt the user to select an option to proceed."""
    logging.info("Please select an option to proceed:")
    logging.info("1: Submit a YouTube Video Link")
    logging.info("2: Use an existing video file")
    return input("Please choose either option 1 or 2: ")


def process_choice(choice) -> Optional[bool]:
    """Process the user choice."""
    if choice == '1':
        logging.info("Submitting a YouTube Video Link")
        url = input("Enter the YouTube URL: ")
        ytdl_main(url, FOLDERS['input'], FOLDERS['whisper_output'], FOLDERS['whisper_output'])
        return False
    elif choice == '2':
        if os.listdir(FOLDERS['input']):
            logging.info("Using an existing video file")
            return True
        else:
            logging.error(f"No video files found in the folder: {FOLDERS['input']}")
            return None
    else:
        logging.error("Invalid choice")
        return None


def create_directories() -> None:
    """Create necessary directories."""
    try:
        for folder in FOLDERS.values():
            os.makedirs(folder, exist_ok=True)
    except Exception as e:
        logging.error(f"Error creating directories: {e}")
        raise


def main():
    required_vars = ['OPENAI_API_KEY', 'GEMINI_API_KEY']
    check_required_vars(required_vars)

    create_directories()

    while True:
        choice = get_user_choice()
        transcribe_flag = process_choice(choice)
        if transcribe_flag is not None:
            break

    local_whisper_process(FOLDERS['input'], FOLDERS['output_video'], FOLDERS['crew_output'], transcribe_flag=transcribe_flag)


if __name__ == "__main__":
    main()