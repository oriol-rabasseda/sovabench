import os
import os.path as osp
import cv2


def generate_video_thumbnail(video_path: str, frame_filename: str, out_dir: str, max_frame_size=-1) -> str | cv2.Mat:
    """
    Generate a thumbnail for a video
    Args:
        camera_name: name of the camera, like "school_G474"
        video_path: path to the video file
        out_dir: path to the output directory
        max_frame_size: maximum width or height of the frame. If -1, the frame will not be resized
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file: " + video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2

    # Set position to middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

    # Read the middle frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read frame at middle of video: " + video_path)

    # Resize frame to have a maximum side length of 224 pixels
    height, width = frame.shape[:2]

    scale = max_frame_size / max(height, width) if max_frame_size > 0 else 1
    new_size = (int(width * scale), int(height * scale))
    resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # Save and return the frame file name
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(osp.join(out_dir, frame_filename), resized_frame)
    return frame_filename


def get_video_info(video_filepath: str) -> dict:
    """
    Returns a dictionary containing:
        - 'duration': float, in seconds
        - 'width': int
        - 'height': int
        - 'fps': float
        - 'frame_count': int
    """
    cap = cv2.VideoCapture(video_filepath)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_filepath}")

    # Get metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        raise ValueError(f"Invalid FPS value ({fps}) in video {video_filepath}")

    duration = frame_count / fps

    cap.release()

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": round(fps),
        "frame_count": frame_count
    }


def crop_and_save_clip(video_filepath: str, out_video_filepath: str, bbox: list, min_crop_size: int) -> bool:
    """
    Crop a video based on bounding box coordinates and save the result.
    Args:
        video_filepath (str): Path to input video file
        out_video_filepath (str): Path to output video file
        bbox (list): Bounding box coordinates [x1, y1, x2, y2]; (x2, y2) is exclusive according to the CV convention
    Returns:
        bool: True if successful, False otherwise
    """

    # Validate input
    if not os.path.exists(video_filepath):
        print(f"Error: Video file not found: {video_filepath}")
        return False

    if len(bbox) != 4:
        print("Error: bbox must contain exactly 4 values [x1, y1, x2, y2]")
        return False

    x1, y1, x2, y2 = bbox

    # Open video file
    cap = cv2.VideoCapture(video_filepath)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_filepath}")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Validate bounding box coordinates
    # Force x1, y1, x2, y2 to range [0, width] and [0, height]
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x1 >= x2 or y1 >= y2:
        print(f"Error: Invalid bounding box coordinates {[{x1}, {y1}, {x2}, {y2}]}: {video_filepath}")
        cap.release()
        return False

    # Calculate cropped dimensions
    crop_width = x2 - x1
    crop_height = y2 - y1    

    if crop_width < min_crop_size or crop_height < min_crop_size:
        cap.release()
        if osp.exists(out_video_filepath):
            print("Small crop, removing file: " + out_video_filepath)
            os.remove(out_video_filepath)
        return False
    
    if osp.exists(out_video_filepath):
        cap.release()
        return True

    # Create output directory if it doesn't exist
    output_dir = osp.dirname(out_video_filepath)
    os.makedirs(output_dir, exist_ok=True)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec as needed
    out = cv2.VideoWriter(out_video_filepath, fourcc, fps, (crop_width, crop_height))

    if not out.isOpened():
        print(f"Error: Could not create output video file: {out_video_filepath}")
        cap.release()
        return False

    # Process frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[y1:y2, x1:x2]

        # Write the cropped frame
        out.write(cropped_frame)
        frame_count += 1

    # Release everything
    cap.release()
    out.release()

    return True