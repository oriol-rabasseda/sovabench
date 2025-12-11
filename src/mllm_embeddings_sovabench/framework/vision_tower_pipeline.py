import os
import pandas as pd
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
from utils import *
import argparse
import torch


def obtain_embeddings(question_path, base_path, output_folder, model_name, fps):
    df = pd.read_csv(os.path.join(base_path, question_path), sep="\t")

    model = AutoModel.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    output_path = os.path.join(base_path, output_folder)
    os.makedirs(output_path, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        index = row["index"]
        output_file = os.path.join(output_path, f"{index}.npy")

        if os.path.exists(output_file):
            continue

        relative_path = os.path.join(row["prefix"], row["video"] + row["suffix"])
        video_path = os.path.join(base_path, relative_path)
        frame_list = video_to_frames(video_path, fps)

        with torch.no_grad():
            inputs = processor(images=frame_list, return_tensors="pt").to(model.device)
            all_embeddings = model.get_image_features(**inputs)

        mean_emb = all_embeddings.cpu().numpy().mean(0)
        np.save(output_file, mean_emb)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings for videos")

    parser.add_argument("--question_path", type=str, required=True,
                        help="Path to TSV file containing video metadata.")

    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second for sampling.")

    parser.add_argument("--model_name", type=str, default="google/siglip2-giant-opt-patch16-384",
                        help="SentenceTransformer model name.")

    parser.add_argument("--output_folder", type=str, required=True,
                        help="Where to store the .npy embedding files.")
    
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base path before prefix"
    )

    args = parser.parse_args()

    obtain_embeddings(
        question_path=args.question_path,
        model_name=args.model_name,
        fps=args.fps,
        output_folder=args.output_folder,
        base_path=args.base_path.rstrip("/")
    )

if __name__ == "__main__":
    main()