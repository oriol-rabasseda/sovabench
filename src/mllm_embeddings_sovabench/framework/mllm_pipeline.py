import os
import pandas as pd
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModel, Qwen3VLMoeForConditionalGeneration
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from utils import *
import json
import argparse

generation_kwargs = {'max_new_tokens': 1024,
                     'do_sample': False,
                     'num_beams': 1,
                     'top_k': None,
                     'temperature': None,
                     'top_p': None,
}

def generate(instruction, processor, model, frame_list = None, system_prompt=""):
    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                ],
            }
        ]

    if frame_list:
        messages[0]["content"].insert(0, {"type": "video", "video": frame_list})

    if system_prompt:
        messages.insert(0, {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        })

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to(model.device, dtype=model.dtype)
    generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    return generated_ids_trimmed


def generate_minicpm(frame_list, instruction, processor, model, system_prompt=""):
    msgs = [{'role': 'user', 'content': frame_list + [instruction]}]

    if system_prompt:
        msgs.insert(0, {'role': 'system', 'content': [system_prompt]})

    images = []
    for i, msg in enumerate(msgs):
        content = msg["content"]
        cur_msgs = []
        for c in content:
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("(<image>./</image>)")
            elif isinstance(c, str):
                cur_msgs.append(c)
        msg["content"] = "\n".join(cur_msgs)

    text = processor.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[frame_list],
        padding=True,
        return_tensors="pt",
        max_slice_nums=1,
    )

    del inputs['image_sizes']
    inputs = inputs.to(model.device, dtype=model.dtype)
    generated_ids = model.generate(**inputs, **generation_kwargs, tokenizer=processor.tokenizer)
    return generated_ids


def obtain_embeddings(question_path, base_path, output_filepath, model_name, instruction, fps, system_prompt):
    df = pd.read_csv(os.path.join(base_path, question_path), sep="\t")

    if "MiniCPM-V" in model_name:
        model = AutoModel.from_pretrained(
            model_name, torch_dtype="auto", attn_implementation="flash_attention_2", device_map="auto", trust_remote_code=True
        )
    elif "Qwen3-VL" in model_name:
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto", trust_remote_code=True
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
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
        frame_list = video_to_frames(video_path, fps)

        if "MiniCPM-V" in model_name:
            generated_ids_trimmed = generate_minicpm(frame_list, instruction, processor, model, system_prompt)
        else:
            generated_ids_trimmed = generate(instruction, processor, model, frame_list, system_prompt)
        out = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        embeddings[index] = out[0]

        with open(output_path, 'w') as f:
            json.dump(embeddings, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings for videos")

    parser.add_argument("--question_path", type=str, required=True,
                        help="Path to TSV file containing video metadata.")

    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second for sampling.")

    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B-HF",
                        help="HuggingFace's model name.")

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