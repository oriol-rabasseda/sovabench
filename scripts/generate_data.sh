

python ./src/mllm_embeddings_sovabench/datasets/add_distractors.py \
    --retrieval-paths ./datasets/MEVA/generated_clips ./datasets/VIRAT/generated_clips \
    --question-path ./datasets/sovabench_interpair_queries.tsv \
    --output-path ./datasets/sovabench_interpair.tsv \
    --base-path ../