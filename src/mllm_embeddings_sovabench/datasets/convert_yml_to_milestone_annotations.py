"""
Convert DIVA Phase 2 YAML annotations (from MEVA dataset and VIRAT 2020 datasets) to Milestone JSON format.
"""
import os
import os.path as osp
import glob
from tqdm import tqdm
from utils.activity_filering import filter_out_unwanted_activities, convert_virat_to_meva_labels_inplace
from utils.meva_fileutils import parse_filename
from utils.meva_video_annotations import MevaVideoAnnotation, MevaAnnotationEncoder
import argparse

# Activities to delete from the Milestone annotations (all VIRAT specific)
ACTIVITIES_TO_DELETE = frozenset({})


def convert_to_milestone_annotation(geom_filepath: str, dataset_name: str) -> MevaVideoAnnotation:
    # Step 1: Load the raw YAML annotations
    ann = MevaVideoAnnotation()
    ann.import_yaml(geom_filepath)

    # Step 2: Filter out unwanted activities
    ann = filter_out_unwanted_activities(ann, ACTIVITIES_TO_DELETE)

    # Step 3: Convert from VIRAT naming scheme to MEVA naming scheme
    if dataset_name == "VIRAT":
        convert_virat_to_meva_labels_inplace(ann)

    return ann


def convert_to_milestone_annotations(annotation_rootdir: str, ms_ann_dir: str) -> None:
    # Find all *.geom.yml files in the annotation directory and subdirectories
    pattern = osp.join(annotation_rootdir, "**", "*.geom.yml")
    geom_filepaths = sorted(glob.glob(pattern, recursive=True))

    for geom_filepath in tqdm(geom_filepaths):
        video_basename = osp.basename(geom_filepath).replace(".geom.yml", "")

        file_info = parse_filename(video_basename)
        subdir = file_info["date"]

        json_ann_file = osp.join(ms_ann_dir, subdir, video_basename + ".json")
        if osp.exists(json_ann_file):  # Already converted
            continue

        # Step 1: Load the raw YAML annotations
        ann = MevaVideoAnnotation()
        ann.import_yaml(geom_filepath)

        # Step 2: Filter out unwanted activities
        ann = filter_out_unwanted_activities(ann, ACTIVITIES_TO_DELETE)

        # Step 3: Convert from VIRAT naming scheme to MEVA naming scheme
        if file_info["dataset"] == "VIRAT":
            convert_virat_to_meva_labels_inplace(ann)

        # Step 4: Save the milestone annotations
        os.makedirs(osp.dirname(json_ann_file), exist_ok=True)
        ann.write_to_json(json_ann_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert dataset annotations into Milestone format."
    )

    parser.add_argument(
        "--annotation_rootdir",
        type=str,
        required=True,
        help="Path to the root directory containing original annotations."
    )

    parser.add_argument(
        "--ms_ann_dir",
        type=str,
        required=True,
        help="Output directory where Milestone-style annotations will be saved."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    annotation_rootdir = os.path.abspath(args.annotation_rootdir)
    ms_ann_dir = os.path.abspath(args.ms_ann_dir)

    convert_to_milestone_annotations(annotation_rootdir, ms_ann_dir)
