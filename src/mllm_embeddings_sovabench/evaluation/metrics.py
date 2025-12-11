import os
import argparse
from utils import load_embeddings, mean_average_precision

def parse_args():
    parser = argparse.ArgumentParser(description="Compute mAP for GorB benchmark.")

    parser.add_argument(
        "--embedder",
        type=str,
        default="Alibaba-NLP/gte-large-en-v1.5", #'clip-ViT-L-14' 'all-MiniLM-L6-v2' 'Alibaba-NLP/gte-large-en-v1.5' 'google/embeddinggemma-300m' 'jinaai/jina-embedding-l-en-v1' 'Qwen/Qwen3-Embedding-0.6B' 'intfloat/multilingual-e5-large-instruct'
        help="Embedding model used to produce vectors."
    )

    parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="JSON file containing stored embeddings."
    )

    parser.add_argument(
        "--multi-sentence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multi-sentence embeddings (default: True)."
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["binary", "easy", "twoclass"],
        default="easy",
        help="Evaluation method: binary, easy, or twoclass."
    )

    parser.add_argument(
        "--question-path",
        type=str,
        required=True,
        help="Path to the TSV file containing benchmark questions."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    embeddings = load_embeddings(
        args.results_file,
        args.embedder,
        args.multi_sentence
    )

    map_ = mean_average_precision(
        embeddings,
        args.question_path,
        method=args.method
    )

    print(map_)