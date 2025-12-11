import os
import os.path as osp
import json
from glob import glob
from tqdm import tqdm
from utils.meva_fileutils import find_first_file, save_clips_by_frame_ranges, add_buffer_to_frame_range
from utils.meva_activity_mapping import MevaActivityMapping
from utils.meva_video_annotations import MevaVideoAnnotation
from utils.video_utils import crop_and_save_clip
from multiprocessing import Pool
from functools import partial
import argparse

def cut_and_save_clips(video_filepath: str,
                        clip_videos_outdir: str,
                        clips_ann_outdir: str,
                        video_act_data: dict) -> None:
    # Extract clips excluding augmented activities
    save_clips_by_frame_ranges(video_filepath,
                                  clip_videos_outdir,
                                  [act["name"] for act in video_act_data],
                                  [act["timespan"] for act in video_act_data])

    # Save the clip annotations
    video_basename = osp.splitext(osp.basename(video_filepath))[0]
    os.makedirs(clips_ann_outdir, exist_ok=True)
    with open(osp.join(clips_ann_outdir, f"{video_basename}.jsonl"), "w") as f:
        for act_data in video_act_data:
            f.write(json.dumps(act_data) + "\n")


def split_one_video(json_path: str, ann_in_dir: str, video_basedir: str, clip_videos_dir: str, clip_ann_dir: str):
    # Find path of json_path relative to in_dir
    rel_dir = osp.relpath(osp.dirname(json_path), ann_in_dir)
    
    video_annotation = MevaVideoAnnotation.read_from_json(json_path)

    # Extract video basename without extension
    video_basename = osp.splitext(osp.basename(json_path))[0]

    # Extract the video filepath
    video_filepath = find_first_file(video_basename, [".avi", ".mp4"], video_basedir)
    if video_filepath is None:
        raise FileNotFoundError(f"Could not find video {video_filepath}")

    # Get target activities; augment activities if applicable
    target_activities = MevaActivityMapping.by_groups(["ignore", "misc", "human_vehicle", "vehicle", "vehicle_augmented"])

    act_data = video_annotation.extract_structured_activities(target_activities)
    #act_data += MevaVideoAnnotation.augment_activities(act_data, target_activities)

    # Extract and save clips
    cut_and_save_clips(video_filepath,
                        osp.join(clip_videos_dir, rel_dir),
                        osp.join(clip_ann_dir, rel_dir),
                        act_data)

def split_videos_by_activities(ann_in_dir: str, video_basedir: str, out_dir: str) -> None:
    """
    Recursively process JSON files and split the videos into clips based on activity.
    """
    clip_videos_dir = osp.join(osp.expanduser(out_dir), "videos_fullsize")
    clip_ann_dir = osp.join(osp.expanduser(out_dir), "annotations")
    os.makedirs(clip_videos_dir, exist_ok=True)
    os.makedirs(clip_ann_dir, exist_ok=True)

    # Recursively find all .json files
    json_files = glob(osp.join(ann_in_dir, "**", "*.json"), recursive=True)
    json_files.sort()

    '''
    pool = Pool(10)
    pool.map(partial(split_one_video, ann_in_dir=ann_in_dir, video_basedir=video_basedir, clip_videos_dir=clip_videos_dir, clip_ann_dir=clip_ann_dir), tqdm(json_files))
    '''
    for json_path in tqdm(json_files):
        split_one_video(json_path, ann_in_dir=ann_in_dir, video_basedir=video_basedir, clip_videos_dir=clip_videos_dir, clip_ann_dir=clip_ann_dir)


def crop_one_video(jsonl_path: list, video_basedir: str, ann_basedir: str, out_dir: str, min_crop_size: int):
    # Find path of json_path relative to in_dir
    rel_dir = osp.relpath(osp.dirname(jsonl_path), ann_basedir)

    # Extract video basename without extension
    video_basename = osp.splitext(osp.basename(jsonl_path))[0]

    activities = []        
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:  # Skip empty lines
                obj = json.loads(line)
                activities.append(obj)

    for act in activities:
        name = act["name"]
        frame_start, frame_end = act["timespan"]
        add_buffer_to_frame_range(name, frame_start, frame_end, 300)
        bbox = act["bbox"]

        video_filepath = osp.join(video_basedir, rel_dir, f"{video_basename}.{name}.{frame_start}.{frame_end}.mp4")
        out_video_filepath = osp.join(out_dir, rel_dir, f"{video_basename}.{name}.{frame_start}.{frame_end}.mp4")
        crop_and_save_clip(video_filepath, out_video_filepath, bbox, min_crop_size)


def crop_videos(video_basedir: str, ann_basedir: str, out_dir: str, min_crop_size: int):
    # Recursively find all .jsonl files
    jsonl_files = glob(osp.join(ann_basedir, "**", "*.jsonl"), recursive=True)
    jsonl_files.sort()

    '''
    pool = Pool(10)
    pool.map(partial(crop_one_video, video_basedir=video_basedir, ann_basedir=ann_basedir, out_dir=out_dir, min_crop_size=min_crop_size), jsonl_files)
    '''
    for jsonl_path in tqdm(jsonl_files):
        crop_one_video(jsonl_path=jsonl_path, video_basedir=video_basedir, ann_basedir=ann_basedir, out_dir=out_dir, min_crop_size=min_crop_size)


def extract_activity_clips(input_dir: str, video_dir: str, out_dir: str):
    # Step 1: Split videos by activities
    split_videos_by_activities(input_dir, video_dir, out_dir)

    # Step 2: Crop the split videos containing the relevant objects
    crop_videos(out_dir + "/videos_fullsize", out_dir + "/annotations", out_dir + "/videos", min_crop_size=112)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract (clip and crop) video clips from activities."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the Milestone-style annotated data."
    )

    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Path to the raw video data."
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory where cropped and clipped videos will be stored."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    video_dir = os.path.abspath(args.video_dir)
    out_dir = os.path.abspath(args.out_dir)
    
    extract_activity_clips(input_dir, video_dir, out_dir)
