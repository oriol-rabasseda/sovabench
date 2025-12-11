import os
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from utils import *
import json
from transformers import AutoModelForCausalLM, AutoProcessor
import argparse

generation_kwargs = {'max_new_tokens': 1024,
                     'do_sample': False,
                     'num_beams': 1,
                     'top_k': None,
                     'temperature': None,
                     'top_p': None,
}


def obtain_embeddings(question_path, base_path, output_filepath, model_name, instruction, fps, system_prompt, min_frames=2):
    df = pd.read_csv(os.path.join(base_path, question_path), sep="\t")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
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

        if system_prompt:
            conversation = [
                {"role": "system", "content": system_prompt}
            ]
        else:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

        conversation.append({
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "max_frames": nframes}},
                {"type": "text", "text": instruction},
            ]
        })

        inputs = processor(conversation=conversation, return_tensors="pt")

        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = model.generate(**inputs, **generation_kwargs)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        embeddings[index] = response

        with open(output_path, 'w') as f:
            json.dump(embeddings, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings for videos")

    parser.add_argument("--question_path", type=str, required=True,
                        help="Path to TSV file containing video metadata.")

    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second for sampling.")

    parser.add_argument("--model_name", type=str, default="DAMO-NLP-SG/VideoLLaMA3-7B",
                        help="Model name of the family Video-Llama.")

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