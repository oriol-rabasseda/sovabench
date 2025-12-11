import os
import os.path as osp
from pathlib import Path
from fnmatch import fnmatch
import cv2
import csv

# Buffer frame counts (before, after) for different types of video clips.
# If not undefined, there is no buffer
CLIP_BUFFERS = {
    "vehicle_starts": (90, 0),
    "vehicle_stops": (0, 90)
}


def parse_filename(filename: str) -> dict:
    """
    Parse a structured filename into its components.
    The supported filename formats are:
        - Base filename: e.g., `2018-03-05.13-15-00.13-20-00.bus.G340`
        - Video filename: e.g., `2018-03-05.09-49-37.09-50-00.school.G474.r13.avi`
        - Annotation filename: e.g., `2018-03-05.13-15-00.13-20-00.bus.G340.geom.yml`
    """
    parts = filename.split(".")
    fileinfo = {}

    if len(parts) >= 5:  # MEVA
        if len(parts) >= 9:
            fileinfo["type"] = "clip"  # Clip for one activity
            fileinfo["activity"] = parts[6]
        elif len(parts) >= 6:
            if parts[-1] == "avi":
                fileinfo["type"] = "video"
                fileinfo["session"] = parts[5]
            elif parts[-1] == "yml":
                fileinfo["type"] = "annotation"
            else:
                raise ValueError(f"Unknown file type: {parts[-1]}")
        elif len(parts) == 5:
            fileinfo["type"] = "base"

        fileinfo.update({"dataset": "MEVA",
                         "date": parts[0],
                         "start_time": parts[1].replace("-", ":"),
                         "end_time": parts[2].replace("-", ":"),
                         "location": parts[3],
                         "camera_id": parts[4]})
    elif 1 <= len(parts) <= 2:  # VIRAT 2020
        subparts = parts[0].split("_")
        fileinfo.update({"dataset": "VIRAT",
                         "camera_id": "_".join(subparts[:2]) + "_" + subparts[2][:4],  # Like VIRAT_S_0002
                         "date": ""  # Not provided
                         })
    else:
        raise ValueError(f"Unknown filename format: {filename}")

    return fileinfo


def get_ann_filepath(ann_root: str, clip_basename: str) -> str | None:
    """
    Get the annotation file path for a given clip name
    Args:
        ann_root (str): The root directory of the MEVA data annotation repository, such as /my-parent-path/meva-data-repo/
        clip_name (str): The base filename of the clip, such as "2018-03-05.13-15-00.13-20-00.bus.G340"
    Returns:
        str: The path to the annotation file. If not found, returns None.
    """
    meva_ann_dir = osp.join(ann_root, "annotation/DIVA-phase-2/MEVA/")

    ann_subdirs = ["kitware", "kitware-meva-training"]  # Different annotation batches
    fileinfo = parse_filename(clip_basename)
    if fileinfo["type"] != "base":
        return ValueError(f"Not a base file: {clip_basename}")

    for ann_subdir in ann_subdirs:
        date_dir = osp.join(meva_ann_dir, ann_subdir, fileinfo["date"])
        filename = clip_basename + ".geom.yml"

        # Go through all subdirs, and check if the annotation file exists
        session_dirs = [dir for dir in os.listdir(date_dir) if osp.isdir(osp.join(date_dir, dir))]
        ann_filepath = next((osp.join(date_dir, session_dir, filename) for session_dir in session_dirs
                             if osp.exists(osp.join(date_dir, session_dir, filename))), None)
        if ann_filepath is not None:
            print(f"Found annotation in {ann_subdir}")
            return ann_filepath

    return None


def find_first_file(basename: str, extensions: str | list[str], rootdir: str) -> str | None:
    """
    Search recursively in `rootdir` for the first file matching `basename*.<ext>` for any extension in `extensions`.

    Args:
        basename: Base name prefix (e.g., "video")
        extensions: A single extension (e.g., ".mp4") or a list/tuple of extensions (e.g., [".avi", ".mp4"])
        rootdir: Root directory to start search from

    Returns:
        str: Full path to the first matching file found.
        None: If no matching file is found.
    """
    if isinstance(extensions, str):
        extensions = [extensions]

    patterns = [f"{basename}*{ext}" for ext in extensions]
    for root, _, files in os.walk(rootdir):
        for file in files:
            for pattern in patterns:
                if fnmatch(file, pattern):
                    return osp.join(root, file)
    return None


def get_video_filepath(video_root: str, clip_basename: str) -> str | None:
    """
    Get the video file path for a given clip name
    Args:
        video_root (str): The root directory of the MEVA data annotation repository, such as /my-parent-path/videos/
        clip_basename (str): The base name of the clip, such as "2018-03-05.13-15-00.13-20-00.bus.G340"
    Returns:
        str: The path to the video file. If not found, returns None.
    """
    fileinfo = parse_filename(clip_basename)
    if fileinfo["type"] != "base":
        raise ValueError(f"Not a base file: {clip_basename}")

    root = Path(video_root)
    date_dir = root / fileinfo["date"]
    video_filepaths = list(date_dir.rglob(f"{clip_basename}.*.avi"))
    if len(video_filepaths) > 1:
        raise ValueError(f"Multiple video files found for {clip_basename}")

    return str(video_filepaths[0]) if len(video_filepaths) > 0 else None


def add_buffer_to_frame_range(act_name: str, start_frame: int, end_frame: int, total_frames: int) -> tuple[int, int]:
    buffer = CLIP_BUFFERS.get(act_name, (0, 0))
    sign = -1 if start_frame > end_frame else 1

    return (max(0, min(total_frames - 1, start_frame - sign * buffer[0])),
            max(0, min(total_frames - 1, end_frame + sign * buffer[1])))


def save_clips_by_frame_ranges(video_path: str, output_dir: str, act_names: list[str], frame_ranges: list[tuple[int, int]]) -> None:
    """
    Extract multiple video clips from specified frame ranges and save them as separate files.
    Supports reversed clips (start_frame > end_frame) by playing frames in reverse order.
    """
    if not act_names:
        return

    if len(act_names) != len(frame_ranges):
        raise ValueError("Number of activities and frame ranges must be the same.")

    os.makedirs(output_dir, exist_ok=True)
    video_basename = osp.splitext(osp.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Validate all frame ranges
        for idx, (start, end) in enumerate(frame_ranges):
            if start < 0 or end < 0 or start >= total_frames or end >= total_frames:
                raise ValueError(f"Invalid frame range {frame_ranges[idx]} for activity {act_names[idx]}")

        for (start_frame, end_frame), act_name in zip(frame_ranges, act_names):
            _save_single_clip(cap, output_dir, video_basename, act_name, start_frame,
                              end_frame, fps, width, height, total_frames)
    finally:
        cap.release()


def _save_single_clip(cap: cv2.VideoCapture, output_dir: str, video_basename: str, act_name: str, start_frame: int,
                      end_frame: int, fps: float, width: int, height: int, total_frames: int) -> None:
    output_path = osp.join(output_dir, f"{video_basename}.{act_name}.{start_frame}.{end_frame}.mp4")
    if osp.exists(output_path):
        return
    
    start_frame, end_frame = add_buffer_to_frame_range(act_name, start_frame, end_frame, total_frames)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not out.isOpened():
        out.release()
        raise IOError(f"Failed to open VideoWriter for: {output_path}")

    try:
        if start_frame > end_frame:  # Reversed
            logical_start = end_frame
            logical_end = start_frame
        else:
            logical_start = start_frame
            logical_end = end_frame

        frames = []
        current = logical_start
        while current <= logical_end:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current)
            ret, frame = cap.read()
            if not ret:
                raise IOError(f"Failed to read frame {current}.")
            frames.append(frame)
            current += 1

        if start_frame > end_frame:  # Reversed
            frames = reversed(frames)

        for frame in frames:
            out.write(frame)
    finally:
        out.release()


def write_to_tsv(
    output_tsv: str,
    qa_list: list[dict],
    task_type: str,
    mode: str = "overwrite",  # "overwrite" or "append"
    prefix: str = "./videos"
) -> None:
    """
    Write QA data to a TSV file.
    """
    if mode not in {"overwrite", "append"}:
        raise ValueError("mode must be 'overwrite' or 'append'")

    # Map mode to file open mode
    file_mode = 'w' if mode == "overwrite" else 'a'

    # Decide whether to write header:
    # - Always write if overwriting
    # - Write only if file doesn't exist yet when appending
    write_header = mode == "overwrite" or (mode == "append" and not osp.exists(output_tsv))

    with open(output_tsv, file_mode, encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f,
                                fieldnames=[
                                    "task_type", "prefix", "suffix", "video",
                                    "positive", "negative", "dim", "subdim", "index"
                                ],
                                delimiter='\t')

        if write_header:
            writer.writeheader()

        start_index = 0
        if mode == "append" and osp.exists(output_tsv) and osp.getsize(output_tsv) > 0:
            # Read last index
            with open(output_tsv, 'r', encoding='utf-8') as existing_f:
                lines = existing_f.readlines()
                if len(lines) > 1:  # Has header + at least one row
                    last_line = lines[-1]
                    last_index = int(last_line.split('\t')[-1])  # assumes 'index' is last column
                    start_index = last_index + 1

        for i, qa in enumerate(qa_list):
            if 'VIRAT' in output_tsv:
                prefix_aux = prefix
            else:
                date_dir = qa["video"].split(".")[0]
                prefix_aux = os.path.join(prefix, date_dir)
            
            row = {
                "task_type": task_type,
                "prefix": prefix_aux,
                "suffix": ".mp4",
                "video": qa["video"],
                "positive": qa["positive"],
                "negative": qa["negative"],
                "dim": qa["group"],
                "subdim": qa["subgroup"],
                "index": start_index + i,
            }
            writer.writerow(row)


def merge_tsv_files(input_files: list[str], output_file: str) -> None:
    """
    Merge multiple TSV files into one, aligning by headers.
    If some files are missing columns, those fields are left empty.
    """
    all_headers = set()

    # First pass: collect all headers from all files
    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames:
                all_headers.update(reader.fieldnames)

    # Ensure "index" is always included, and put it last for consistency
    all_headers = [h for h in all_headers if h != "index"] + ["index"]

    with open(output_file, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=all_headers, delimiter="\t")
        writer.writeheader()

        new_index = 0
        for file in input_files:
            with open(file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    # Fill missing fields
                    full_row = {h: row.get(h, "") for h in all_headers}
                    full_row["index"] = new_index  # overwrite with new unique index
                    writer.writerow(full_row)
                    new_index += 1
