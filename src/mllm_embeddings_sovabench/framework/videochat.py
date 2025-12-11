import os
import pandas as pd
from tqdm import tqdm
from utils import *
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import numpy as np
import cv2
from PIL import Image

generation_kwargs = {'max_new_tokens': 1024,
                     'do_sample': False,
                     'num_beams': 1,
                     'top_k': None,
                     'temperature': None,
                     'top_p': None,
}


def obtain_embeddings(question_path, base_path, output_filepath, model_name, instruction, fps, system_prompt, min_frames = 2):
    df = pd.read_csv(os.path.join(base_path, question_path), sep="\t")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto",
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    output_path = os.path.join(base_path, output_filepath)
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            embeddings = json.load(f)
    else:
        embeddings = dict()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        index = row["index"]
        if str(index) in embeddings:
            continue
        
        relative_path = os.path.join(row["prefix"], row["video"] + row["suffix"])
        video_path = os.path.join(base_path, relative_path)

        video = cv2.VideoCapture(video_path)
        fps_video = round(video.get(cv2.CAP_PROP_FPS))
        frame_count = round(video.get(cv2.CAP_PROP_FRAME_COUNT))
        nframes = round(frame_count/fps_video*fps) + 1
        if nframes < min_frames:
            nframes = min_frames
        
        image_list_idx = np.rint(np.linspace(0, frame_count-1, nframes))

        i = 0
        j = 0
        frame_list = []
        while(video.isOpened()):
            ret, frame = video.read()
            if ret == False:
                break
            
            if i == int(image_list_idx[j]):
                frame_list.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

                j += 1
                if j >= len(image_list_idx):
                    break
            i += 1
        video.release()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frame_list,
                        "max_pixels": 460800,
                        "nframes": nframes
                    },
                    {"type": "text", "text": f"""{instruction}
                    Provide your final answer within the <answer> </answer> tags.
                    """},
                ],
            }
        ]

        if system_prompt:
            messages.insert(0, {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            })

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        video_kwargs['fps'] = [fps]

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(model.device, dtype=model.dtype)

        # Inference
        generated_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        out = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        embeddings[index] = out[0][8:-9]

        with open(output_path, 'w') as f:
            json.dump(embeddings, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings for videos")

    parser.add_argument("--question_path", type=str, required=True,
                        help="Path to TSV file containing video metadata.")

    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second for sampling.")

    parser.add_argument("--model_name", type=str, default="OpenGVLab/VideoChat-R1_7B",
                        help="Model name.")

    parser.add_argument("--output_filepath", type=str, required=True,
                        help="Where to store the responses.")
    
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base path before prefix"
    )

    parser.add_argument("--instruction", type=str, default="Describe this video.",
                        help="Instruction to the model."
    )

    parser.add_argument("--system-prompt", type=str, default="",
                        help="System prompt."
    )

    args = parser.parse_args()

    obtain_embeddings(
        question_path=args.question_path,
        model_name=args.model_name,
        fps=args.fps,
        output_filepath=args.output_filepath,
        base_path=args.base_path.rstrip("/"),
        instruction=args.instruction,
        system_prompt=args.system_prompt
    )

if __name__ == "__main__":
    main()