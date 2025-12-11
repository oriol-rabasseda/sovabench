# Supported Models
## Contrastive Vision-Language Models
To run SOVABench on VLMs, use:
```bash
python vision_tower_pipeline.py \
    --question_path mllm_embedding/datasets/sovabench_interpair.tsv \ # Path to the SOVABench samples
    --fps 1 \ # Number of FPS for sampling rate
    --output_folder mllm_embedding/outputs/clip \ # Output folder to store embeddings
    --base_path your/root/path \
    --model_name openai/clip-vit-base-patch32 \ # Model name from HuggingFace
```

Supported models must be available in [🤗](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification). Evaluated models are:

| Model | Link |
| ----- | ---- |
| CLIP-ViT-B/32 | [🤗 link](https://huggingface.co/openai/clip-vit-base-patch32) |
| SigLIP2-Giant | [🤗 link](google/siglip2-giant-opt-patch16-384) |

---

## General MLLMs
General MLLMs can be run using:
```bash
python mllm_pipeline.py \
    --question_path mllm_embedding/datasets/sovabench_interpair.tsv \ # Path to the SOVABench samples
    --fps 1 \ # Number of FPS for sampling rate
    --output_filepath mllm_embedding/outputs/minicpmv_4_5.json \ # Output JSON filepath
    --base_path your/root/path \
    --model_name openbmb/MiniCPM-V-4_5 \ # Model name from HuggingFace
    --instruction "Describe this video." \ # User instruction to the MLLM
    --system-prompt "" # System instruction to the MLLM, if empty, uses default system prompting strategy
```

The set of supported models is:

| Model | Link |
| ----- | ---- |
| InternVL 3.5 | [🤗 link](https://huggingface.co/collections/OpenGVLab/internvl35) |
| MiniCPM-V 4.5 | [🤗 link](https://huggingface.co/openbmb/MiniCPM-V-4_5) |

---

## Video MLLMs
To run Video-LLaVA ([🤗 link](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)):
```bash
python videollava.py \
    --question_path mllm_embedding/datasets/sovabench_interpair.tsv \ # Path to the SOVABench samples
    --fps 1 \ # Number of FPS for sampling rate
    --output_filepath mllm_embedding/outputs/videollava.json \ # Output JSON filepath
    --base_path your/root/path \
    --instruction "Describe this video." \ # User instruction to the MLLM
    --system-prompt "" # System instruction to the MLLM, if empty, uses default system prompting strategy
```

To run Video-Llama3 ([🤗 link](https://huggingface.co/collections/DAMO-NLP-SG/videollama3)):
```bash
python videollama.py \
    --question_path mllm_embedding/datasets/sovabench_interpair.tsv \ # Path to the SOVABench samples
    --fps 1 \ # Number of FPS for sampling rate
    --output_filepath mllm_embedding/outputs/videollama.json \ # Output JSON filepath
    --base_path your/root/path \
    --model_name DAMO-NLP-SG/VideoLLaMA3-7B \ # Model name from HuggingFace
    --instruction "Describe this video." \ # User instruction to the MLLM
    --system-prompt "" # System instruction to the MLLM, if empty, uses default system prompting strategy
```

To run Video-Chat R1 ([🤗 link](https://huggingface.co/OpenGVLab/VideoChat-R1_7B)):
```bash
python videochat.py \
    --question_path mllm_embedding/datasets/sovabench_interpair.tsv \ # Path to the SOVABench samples
    --fps 1 \ # Number of FPS for sampling rate
    --output_filepath mllm_embedding/outputs/videollama.json \ # Output JSON filepath
    --base_path your/root/path \
    --instruction "Describe this video." \ # User instruction to the MLLM
    --system-prompt "" # System instruction to the MLLM, if empty, uses default system prompting strategy
```

---

# Hyperparameters used
To reproduce the results reported on the original paper, the parameters used were:
- Sampling rate: 1 FPS
- User instruction: Briefly classify the actions occurring in this video.
- System instruction: You are an expert video analysis model specialized in action recognition. Focus on how subjects and objects change and move over time rather than on static appearances or backgrounds. Infer the actions by reasoning about motion, temporal progression, and interactions across the video frames.