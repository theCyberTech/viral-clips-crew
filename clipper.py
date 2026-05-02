# Standard library imports
import os
import warnings
import re
from datetime import datetime
import glob
import logging

# Third party imports
import ffmpeg

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore")


def convert_timestamp(timestamp):
    return timestamp.replace(',', '.').strip()


def parse_timestamp(timestamp):
    return datetime.strptime(timestamp, '%H:%M:%S.%f')


def get_aspect_ratio_choice():
    while True:
        choice = input("Choose aspect ratio for all videos: (1) Keep as original, (2) 1:1 (square): ")
        if choice in ['1', '2']:
            return choice
        print("Invalid choice. Please enter 1 or 2.")


def parse_srt_blocks(subtitles_content):
    """
    Parse SRT content into individual subtitle blocks.
    Returns list of (start_time_str, end_time_str) tuples, sorted by start time.
    """
    blocks = []
    raw_blocks = re.split(r'\n\s*\n', subtitles_content.strip())

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        match = re.search(
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', block
        )
        if match:
            start = convert_timestamp(match.group(1))
            end = convert_timestamp(match.group(2))
            blocks.append((start, end))

    # Sort by start time
    blocks.sort(key=lambda b: b[0])
    return blocks


def group_contiguous_blocks(blocks, max_gap_seconds=2.0):
    """
    Group subtitle blocks into contiguous segments.
    Blocks with a gap <= max_gap_seconds are merged into one clip.
    Returns list of (group_start, group_end) tuples.
    """
    if not blocks:
        return []

    groups = []
    current_start, current_end = blocks[0]

    for block_start, block_end in blocks[1:]:
        gap = (parse_timestamp(block_start) - parse_timestamp(current_end)).total_seconds()
        if gap <= max_gap_seconds:
            current_end = block_end
        else:
            groups.append((current_start, current_end))
            current_start, current_end = block_start, block_end

    groups.append((current_start, current_end))
    return groups


def clip_video_segment(input_video, start_time, end_time, output_path, aspect_ratio_choice, segment_index=None):
    """
    Clip a single segment from input_video and save to output_path.
    """
    start_datetime = parse_timestamp(start_time)
    end_datetime = parse_timestamp(end_time)
    duration = end_datetime - start_datetime
    duration_seconds = duration.total_seconds()

    logging.info(f"  Segment {segment_index or ''}: {start_time} --> {end_time} ({duration_seconds:.2f}s)")

    try:
        probe = ffmpeg.probe(input_video)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        input_stream = ffmpeg.input(input_video, ss=start_time, t=duration_seconds)

        if aspect_ratio_choice == '2':  # 1:1 (square)
            if width > height:
                crop_size = height
                x_offset = (width - crop_size) // 2
                y_offset = 0
            else:
                crop_size = width
                x_offset = 0
                y_offset = (height - crop_size) // 2

            video = input_stream.video.filter('crop', crop_size, crop_size, x_offset, y_offset)
        else:
            video = input_stream.video

        audio = input_stream.audio

        output = ffmpeg.output(video, audio, output_path,
                               vcodec='libx264', acodec='aac',
                               audio_bitrate='192k',
                               **{'vsync': 'vfr'})

        ffmpeg.run(output, overwrite_output=True)
        logging.info(f"  Trimmed video saved to {output_path}")
        return True

    except ffmpeg.Error as e:
        logging.error(f"  ffmpeg error for segment {segment_index or ''}: {str(e)}")
        return False


def process_video(input_video, subtitle_file_path, output_folder, aspect_ratio_choice):
    logging.info('~~~CLIPPER: PROCESSING VIDEO~~~')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(subtitle_file_path, 'r') as file:
        subtitles_content = file.read()

    assert subtitles_content != "", "clipper.py received an empty subtitles file"

    # Parse SRT into individual blocks instead of scanning all timestamps globally
    blocks = parse_srt_blocks(subtitles_content)
    if not blocks:
        logging.warning("No subtitle blocks with timestamps found.")
        return

    logging.info(f"Found {len(blocks)} subtitle block(s) in {os.path.basename(subtitle_file_path)}")

    # Group contiguous blocks into clips (gap <= 2s between blocks is treated as same clip)
    clips = group_contiguous_blocks(blocks, max_gap_seconds=2.0)

    subtitle_base_name = os.path.splitext(os.path.basename(subtitle_file_path))[0]

    for i, (start_time, end_time) in enumerate(clips):
        duration_seconds = (parse_timestamp(end_time) - parse_timestamp(start_time)).total_seconds()

        if duration_seconds < 30:
            logging.warning(
                f"  Clip {i+1} duration ({duration_seconds:.2f}s) is less than 30s. Skipping."
            )
            continue
        if duration_seconds > 150:
            logging.warning(
                f"  Clip {i+1} duration ({duration_seconds:.2f}s) exceeds 2m30s. Skipping."
            )
            continue

        if len(clips) > 1:
            output_video_path = os.path.join(output_folder, f"{subtitle_base_name}_trimmed_{i+1}.mp4")
        else:
            output_video_path = os.path.join(output_folder, f"{subtitle_base_name}_trimmed.mp4")

        clip_video_segment(input_video, start_time, end_time, output_video_path,
                           aspect_ratio_choice, segment_index=i + 1)


def main(input_video, subtitle_file_path, output_folder, aspect_ratio_choice=None):
    if aspect_ratio_choice is None:
        aspect_ratio_choice = get_aspect_ratio_choice()
    process_video(input_video, subtitle_file_path, output_folder, aspect_ratio_choice)


if __name__ == "__main__":
    video_files = glob.glob('input_files/*.mp4')
    subtitle_files = glob.glob('crew_output/*.srt')
    output_folder = "clipper_output"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ask for aspect ratio choice once
    aspect_ratio_choice = get_aspect_ratio_choice()

    for video_file_path in video_files:
        for subtitle_file in subtitle_files:
            process_video(video_file_path, subtitle_file, output_folder, aspect_ratio_choice)