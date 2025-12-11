"""
Activity filtering
"""
from collections.abc import Iterable
from .meva_video_annotations import MevaVideoAnnotation
from .meva_activity_mapping import activity_mapping_virat_to_meva


def filter_out_unwanted_activities(annotation: MevaVideoAnnotation, exclude_list: Iterable[str]) -> MevaVideoAnnotation:
    """
    Cleans up the annotation by:
    1. Removing activities listed in exclude_list.
    2. Removing unreferenced objects PER FRAME — only keep object in a frame
       if that (object_id, frame) is covered by at least one remaining activity's timespan.

    Args:
        annotation (dict): The original annotation dict.
        exclude_list (list): List of activity names to exclude.

    Returns:
        dict: Cleaned annotation.
    """
    # Step 1: Filter out excluded activities
    remaining_activities = [act for act in annotation["activities"] if act["name"] not in exclude_list]

    # Step 2: Build frame-to-object usage map
    # For each (frame, obj_id), mark if it's used by any remaining activity
    frame_obj_usage = set()  # set of (frame, obj_id_int)

    for act in remaining_activities:
        actors = act.get("actors", {})
        for obj_id_str, timespans in actors.items():
            obj_id = int(obj_id_str)
            for [start, end] in timespans:
                for frame in range(start, end + 1):  # inclusive end
                    frame_obj_usage.add((frame, obj_id))

    # Step 3: Filter objects — keep only (frame, obj) pairs that are used
    cleaned_objects = {}
    for frame_str, obj_list in annotation["objects"].items():
        frame_num = int(frame_str)
        kept_objs = [obj for obj in obj_list if (frame_num, obj.get("id1")) in frame_obj_usage]
        if kept_objs:
            cleaned_objects[frame_str] = kept_objs

    # Step 4: Return cleaned annotation
    return MevaVideoAnnotation({"activities": remaining_activities, "objects": cleaned_objects})


def convert_virat_to_meva_labels_inplace(ann: MevaVideoAnnotation) -> None:
    """
    Convert activity labels in a MevaVideoAnnotation from VIRAT naming scheme to MEVA naming scheme, in-place.
    """
    for activity in ann["activities"]:
        old_name = activity.get("name")
        if old_name in activity_mapping_virat_to_meva:
            activity["name"] = activity_mapping_virat_to_meva[old_name]