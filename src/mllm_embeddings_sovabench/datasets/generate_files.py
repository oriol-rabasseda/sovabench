import glob
import os.path as osp
from random import Random
import json
from utils.meva_fileutils import write_to_tsv, merge_tsv_files
from utils.meva_activity_mapping import MevaActivityType, MevaActivityMapping
import argparse
import os

def generate_sample(act: MevaActivityType) -> tuple[str, str]:
    act_label = MevaActivityMapping.action_label(act)
    subject_name = MevaActivityMapping.subject_name(act)
    if subject_name is None:
        raise ValueError(f"Cannot generate question for ignored activity: {act.name}")
    
    act_label_ant = MevaActivityMapping.action_label(MevaActivityMapping.antonym(act))
    subject_name_ant = MevaActivityMapping.subject_name(MevaActivityMapping.antonym(act))

    positive = f"{subject_name} {act_label}"
    negative = f"{subject_name_ant} {act_label_ant}"

    return positive, negative


def generate(jsonl_ann_dir: str, video_clip_dir: str, target_acts: set[MevaActivityType]):
    data = []
    jsonl_paths = glob.glob(osp.join(jsonl_ann_dir, "**", "*.jsonl"), recursive=True)
    jsonl_paths.sort()

    clip_paths = glob.glob(osp.join(video_clip_dir, "**", "*.mp4"), recursive=True)
    clip_basenames = {osp.splitext(osp.basename(p))[0] for p in clip_paths}
    
    for jsonl_path in jsonl_paths:
        video_basename = osp.splitext(osp.basename(jsonl_path))[0]

        # Load the activities for this video
        with open(jsonl_path, "r", encoding="utf-8") as f:
            video_acts = [json.loads(line) for line in f if line.strip()]

        # Select activities in the target set
        video_acts = [(i, act) for i, act in enumerate(video_acts) if MevaActivityType(act["name"]) in target_acts]
               
        #video_acts_no_overlapping = [(i, act) for i, act in enumerate(video_acts)]

        for i, act_dict in video_acts:
            clip_basename = f"{video_basename}.{act_dict['name']}.{act_dict['timespan'][0]}.{act_dict['timespan'][1]}"
            if not clip_basename in clip_basenames:
                continue

            # Generate positive sample
            act = MevaActivityType(act_dict["name"])
            pos, neg = generate_sample(act)
            data.append({
                "video": clip_basename,
                "positive": pos,
                "negative": neg,
                "group": MevaActivityMapping.group(act),
                "subgroup": MevaActivityMapping.action_label(act)
            })

    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TSVs for MEVA & VIRAT and merge into a benchmark file."
    )

    parser.add_argument(
        "--meva-dir",
        type=str,
        required=True,
        help="Path to MEVA directory."
    )

    parser.add_argument(
        "--virat-dir",
        type=str,
        required=True,
        help="Path to VIRAT directory."
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to write merged benchmark TSV."
    )

    parser.add_argument(
        "--distractors",
        action="store_true",
        help="Generate distracting samples (default: False)."
    )

    return parser.parse_args()
    

def main():
    args = parse_args()

    input_dir_meva = args.meva_dir
    input_dir_virat = args.virat_dir
    output_file = args.output_file

    if args.distractors:
        act_group = MevaActivityMapping.by_groups(['ignore', 'misc', 'unknown'])
        filenames = ["MEVA_distractors.tsv", "VIRAT_distractors.tsv"]
    else:
        act_group = MevaActivityMapping.by_groups(["human_vehicle", "vehicle", "vehicle_augmented"])
        filenames = ["MEVA_all.tsv", "VIRAT_all.tsv"]

    # MEVA
    samples = generate(
        jsonl_ann_dir=osp.join(input_dir_meva, "annotations"),
        video_clip_dir=osp.join(input_dir_meva, "videos"),
        target_acts=act_group
    )

    absolute_path = osp.abspath(input_dir_meva).split(os.sep)
    idx = absolute_path.index('mllm_embedding')

    write_to_tsv(
        osp.join(input_dir_meva, filenames[0]),
        samples,
        task_type="retrieval",
        mode="overwrite",
        prefix=osp.join(os.sep.join(absolute_path[idx:]), "videos")
    )

    # VIRAT
    samples = generate(
        jsonl_ann_dir=osp.join(input_dir_virat, "annotations"),
        video_clip_dir=osp.join(input_dir_virat, "videos"),
        target_acts=act_group
    )

    absolute_path = osp.abspath(input_dir_virat).split(os.sep)
    idx = absolute_path.index('mllm_embedding')

    write_to_tsv(
        osp.join(input_dir_virat, filenames[1]),
        samples,
        task_type="retrieval",
        mode="overwrite",
        prefix=osp.join(os.sep.join(absolute_path[idx:]), "videos")
    )

    # Merge
    merge_tsv_files([osp.join(input_dir_virat, filenames[1]), osp.join(input_dir_meva, filenames[0])], output_file)



if __name__ == "__main__":
    main()