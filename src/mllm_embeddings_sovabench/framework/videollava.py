import os
import pandas as pd
from tqdm import tqdm
from utils import *
import json
import argparse
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

generation_kwargs = {'max_new_tokens': 1024,
                     'do_sample': False,
                     'num_beams': 1,
                     'top_k': None,
                     'temperature': None,
                     'top_p': None,
}


def obtain_embeddings(question_path, base_path, output_filepath, model_name, instruction, fps, system_prompt):
    df = pd.read_csv(os.path.join(base_path, question_path), sep="\t")

    model = VideoLlavaForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    processor = VideoLlavaProcessor.from_pretrained(model_name)

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

        clip = np.stack(frame_list)
        prompt = f"USER: <video>{instruction} ASSISTANT:"

        if system_prompt:
            prompt = f"SYSTEM: {system_prompt} {prompt}"

        inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device, dtype=model.dtype)
        generated_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        embeddings[index] = out

        with open(output_path, 'w') as f:
            json.dump(embeddings, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings for videos")

    parser.add_argument("--question_path", type=str, required=True,
                        help="Path to TSV file containing video metadata.")

    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second for sampling.")

    parser.add_argument("--model_name", type=str, default="LanguageBind/Video-LLaVA-7B-hf",
                        help="Model name of familiy Video-LLaVA.")

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