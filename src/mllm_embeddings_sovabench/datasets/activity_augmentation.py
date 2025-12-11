import os.path as osp
import cv2
from glob import glob
from tqdm import tqdm
from utils.meva_fileutils import parse_filename
import argparse
import os
import json

ACTIVITY_AUGMENTATION = {
    "vehicle_drives_forward": "vehicle_reverses"  # Augmented from "vehicle_reverses"
}


def get_augmented_video_path(video_filepath: str, activity_augmentation_dict: dict[str, str]) -> str:
    base_filename = osp.basename(video_filepath)

    for k, v in activity_augmentation_dict.items():
        if f".{v}." in base_filename:
            filename = f"{base_filename.replace(f'.{v}.', f'.{k}.')}"
            filename_parts = filename.split('.')
            timespan1 = filename_parts[-3]
            timespan2 = filename_parts[-2]
            filename_parts = filename_parts[:-3] + [timespan2, timespan1] + filename_parts[-1:]
            
            return osp.join(osp.dirname(video_filepath), '.'.join(filename_parts))

    raise ValueError(f"Could not find augmentation config for {video_filepath}")


def reverse_video(video_filepath: str, reverse_video_dict: dict[str, str]) -> None:
    if not osp.exists(video_filepath):
        raise FileNotFoundError(f"Input file not found: {video_filepath}")
    
    # Find output file name
    reversed_video_path = get_augmented_video_path(video_filepath, reverse_video_dict)
    
    # Open the video file
    cap = cv2.VideoCapture(video_filepath)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(reversed_video_path, fourcc, fps, (width, height))

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Release the video capture
    cap.release()

    # Write frames in reverse order
    for frame in reversed(frames):
        out.write(frame)

    # Release the video writer
    out.release()


def reverse_videos(input_dir: str, reverse_video_dict: dict[str, str]) -> list:
    # use glob to get all mp4 files recursively in input_dir
    video_files = glob(f"{input_dir}/**/*.mp4", recursive=True)
    video_files.sort()

    paths = list()
    for video_filepath in tqdm(video_files):
        video_basename = osp.splitext(osp.basename(video_filepath))[0]
        fileinfo = parse_filename(video_basename)
        if fileinfo["type"] == "clip" and fileinfo["activity"] in reverse_video_dict.values():
            reverse_video(video_filepath, reverse_video_dict)
            paths.append(video_filepath)
    
    return paths


def write_jsonl(jsonl_dir: str, reverse_act_dict: dict[str, str], original_paths: list[str]) -> None:
    act_dict = {v: k for k, v in reverse_act_dict.items()}

    for path in original_paths:
        base_path = os.sep.join(osp.splitext(path)[0].split(os.sep)[-2:])
        parts = base_path.split('.')
        jsonl_path = osp.join(jsonl_dir, '.'.join(parts[:-3]) + '.jsonl')

        with open(jsonl_path, "r", encoding="utf-8") as f:
            video_acts = [json.loads(line) for line in f if line.strip()]
        
        for act in video_acts:
            if act['timespan'] == [int(parts[-2]), int(parts[-1])] and act['name'] == parts[-3]:
                new_activity = {'name': act_dict[act['name']],
                                'timespan': [act['timespan'][1], act['timespan'][0]],
                                'bbox': act['bbox'],
                                'act_bboxes': act['act_bboxes']}
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(new_activity, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment the reverse activity to create drive forward activities."
    )

    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="MEVA directory with the cropped and clipped videos."
    )

    parser.add_argument(
        "--jsonl-dir",
        type=str,
        required=True,
        help="MEVA directory with the annotations generated."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    original_paths = reverse_videos(args.video_dir, ACTIVITY_AUGMENTATION)
    write_jsonl(args.jsonl_dir, ACTIVITY_AUGMENTATION, original_paths)