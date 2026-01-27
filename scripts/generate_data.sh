python ./src/mllm_embeddings_sovabench/datasets/convert_yml_to_milestone_annotations.py \
    --annotation_rootdir ./datasets/MEVA/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware \
    --ms_ann_dir ./datasets/MEVA/generated_clips/Milestone_annotations/kitware

python ./src/mllm_embeddings_sovabench/datasets/convert_yml_to_milestone_annotations.py \
    --annotation_rootdir ./datasets/VIRAT/viratannotations/validate \
    --ms_ann_dir ./datasets/VIRAT/generated_clips/Milestone_annotations/validate

python ./src/mllm_embeddings_sovabench/datasets/extract_activity_clips.py \
    --input-dir ./datasets/MEVA/generated_clips/Milestone_annotations/kitware \
    --video-dir ./datasets/MEVA/videos \
    --out-dir ./datasets/MEVA/generated_clips

python ./src/mllm_embeddings_sovabench/datasets/extract_activity_clips.py \
    --input-dir ./datasets/VIRAT/generated_clips/Milestone_annotations/validate \
    --video-dir ./datasets/VIRAT/videos_original \
    --out-dir ./datasets/VIRAT/generated_clips

python ./src/mllm_embeddings_sovabench/datasets/activity_augmentation.py \
    --video-dir ./datasets/MEVA/generated_clips/videos \
    --jsonl-dir ./datasets/MEVA/generated_clips/annotations

python ./src/mllm_embeddings_sovabench/datasets/generate_files.py \
    --meva-dir ./datasets/MEVA/generated_clips \
    --virat-dir ./datasets/VIRAT/generated_clips \
    --output-file ./datasets/sovabench_intrapair.tsv

python ./src/mllm_embeddings_sovabench/datasets/correct_overlapping.py \
    --question-path ./datasets/queries_raw.tsv \
    --output-interpair ./datasets/sovabench_interpair_queries.tsv \
    --output-intrapair ./datasets/queries_raw.tsv \
    --base-path ../

python ./src/mllm_embeddings_sovabench/datasets/add_distractors.py \
    --retrieval-paths ./datasets/MEVA/generated_clips ./datasets/VIRAT/generated_clips \
    --question-path ./datasets/sovabench_interpair_queries.tsv \
    --output-path ./datasets/sovabench_interpair.tsv \
    --base-path ../