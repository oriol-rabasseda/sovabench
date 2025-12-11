import pandas as pd
import json
import os
import cv2
import argparse

DURATION = 10
TO_DELETE = ['person_rides_bicycle', 'vehicle_makes_u_turn']

QUERIES = ['vehicle_starts', 'vehicle_reverses',
           'vehicle_drives_forward', 'person_closes_vehicle_door', 'person_exits_vehicle',
           'person_enters_vehicle', 'vehicle_stops', 'person_opens_vehicle_door',
           'person_closes_trunk', 'person_loads_vehicle', 'vehicle_turns_right',
           'person_opens_trunk', 'vehicle_turns_left', 'person_unloads_vehicle']

def filter_overlap(base_path, df):
    overlapping = list()
    
    annotation_paths = {os.path.join(base_path, row['prefix'].replace('videos', 'annotations'), '.'.join(row['video'].split('.')[:-3]) + '.jsonl') for _, row in df.iterrows()}
    for ann in annotation_paths:    
        with open(ann, 'r', encoding='utf-8') as f:
            distr_annotations = list()
            for _, line in enumerate(f, 1):
                distr_annotations.append(json.loads(line))
        
        annotations = [ann_aux for ann_aux in distr_annotations if ann_aux['name'] in QUERIES]
                
        for ann_line in annotations:   
            for distr_line in distr_annotations:
                timespan1 = min(ann_line['timespan'])
                timespan2 = max(ann_line['timespan'])

                if timespan2 > distr_line['timespan'][0] and timespan1 < distr_line['timespan'][1]:
                    x5 = max(ann_line['bbox'][0], distr_line['bbox'][0])
                    y5 = max(ann_line['bbox'][1], distr_line['bbox'][1])
                    x6 = min(ann_line['bbox'][2], distr_line['bbox'][2])
                    y6 = min(ann_line['bbox'][3], distr_line['bbox'][3])
                    
                    if x5 < x6 and y5 < y6:
                        name = '.'.join(os.path.normpath(ann).split(os.path.sep)[-1].split('.')[:-1])
                        overlapping.append('.'.join([name, distr_line['name'], str(distr_line['timespan'][0]), str(distr_line['timespan'][1])]))

    return set(overlapping)


def collect_video_paths(retrieval_paths):
    """Return all .mp4 video paths under path/videos directories."""
    all_videos = []
    for path in retrieval_paths:
        new_path = os.path.join(path, "videos")
        for dp, dn, filenames in os.walk(new_path):
            for f in filenames:
                if os.path.splitext(f)[1] == ".mp4":
                    all_videos.append(os.path.join(dp, f))
    return all_videos


def filter_non_overlapping_videos(video_paths, overlapping):
    """Filter video paths where base filename is not in overlapping set."""
    non_overlapping = []
    for f in video_paths:
        basename = os.path.splitext(os.path.basename(f))[0]
        if basename not in overlapping:
            non_overlapping.append(f)
    return non_overlapping


def find_prefix(video_name):
    path_parts = os.path.abspath(video_name).split(os.path.sep)
    idx = path_parts.index('mllm_embedding')
    return os.path.sep.join(path_parts[idx:-1])


def build_distractor_dataframe(video_paths):
    """Create new df for distractor videos with required fields."""
    df = pd.DataFrame(video_paths, columns=['video_name'])
    df['negative'] = 'distractor'
    df['positive'] = 'distractor'
    df['subdim'] = 'distractor'
    df['dim'] = 'distractor'
    df['task_type'] = 'retrieval'
    df['prefix'] = df['video_name'].apply(lambda x: find_prefix(x))
    df['video'] = df['video_name'].apply(lambda x: '.'.join(os.path.basename(x).split('.')[:-1]))
    df['suffix'] = df['video_name'].apply(lambda x: '.' + x.split('.')[-1])
    df['index'] = df.index + 2922
    return df.drop(columns=['video_name'])


def main(base_path, question_path, output_path, retrieval_paths, max_duration=30, activities_delete=[]):
    # Load dataset
    df = pd.read_csv(question_path, sep="\t")

    # Step 1: find overlapping annotations
    overlapping = filter_overlap(base_path, df)

    # Step 2: collect distractor video files
    all_videos = collect_video_paths(retrieval_paths)

    # Step 3: filter non-overlapping videos
    non_overlapping_videos = filter_non_overlapping_videos(all_videos, overlapping)

    # Step 4: filter by video length to align with queries
    non_overlapping_short = list()
    for video_path in non_overlapping_videos:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = round(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps
        if duration <= max_duration:
            non_overlapping_short.append(video_path)
        video.release()
    
    # Step 5: filter by activity types
    activities = list()
    non_overlapping_filtered = list()
    for video in non_overlapping_short:
        name = video.split('.')[-4]
        if name not in activities_delete + QUERIES:
            non_overlapping_filtered.append(video)

    # Step 6: build distractor dataframe
    new_df = build_distractor_dataframe(non_overlapping_filtered)

    # Step 7: merge and save
    final_df = pd.concat([df, new_df], ignore_index=True)
    final_df.to_csv(output_path, sep="\t", index=False)

    print(final_df.shape)
    print(f"New samples: {len(non_overlapping_filtered)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Addition of distracting samples to SOVABench (Inter-pair)."
    )

    parser.add_argument(
        "--question-path",
        type=str,
        required=True,
        help="Benchmark data before adding distractors."
    )

    parser.add_argument(
        "--retrieval-paths",
        nargs='+',
        required=True,
        help="List of paths to find distracting samples."
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path of the SOVABench (Inter-pair)."
    )

    parser.add_argument(
        "--base-path",
        type=str,
        required=True,
        help="Base path before the prefix in SOVABench annotations."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(args.base_path, args.question_path, args.output_path, args.retrieval_paths, max_duration=DURATION, activities_delete=TO_DELETE)