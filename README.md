# Warmup-Stable-Merge for SDXL LoRA

An implementation of the **Warmup-Stable-Merge (WSM)** learning rate schedule, specifically tailored for [sd-scripts](https://github.com/kohya-ss/sd-scripts) SDXL LoRAs.

Based on the paper:
> **WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training**  
> https://arxiv.org/abs/2507.17634

## Overview

**WSM** allows us to train with a constant learning rate and simulate different decay schedules by merging checkpoints with different coefficients.

This script takes a set of numbered checkpoints and automatically searches for the best merge window using a validation set.

## Usage

The script looks for `safetensors` files following the pattern `*-stepN.safetensors`, where `N` is an integer.  
This is the naming scheme used by [sd-scripts](https://github.com/kohya-ss/sd-scripts)

### 1. Prepare validation images
Place your validation images and matching `.txt` captions in a single folder:
- `image1.png`, `image1.txt`
- `image2.jpg`, `image2.txt`

### 2. Run the script

```bash
python merge_lora.py \
    --dataset-path /path/to/val_images \
    --ckpt-path /path/to/model.safetensors \
    --lora-dir /path/to/lora_dir \
    --prediction-type epsilon \
```
The best model will be saved to `lora-dir` with a `-merged` suffix.

### Arguments

| Argument | Description |
| :--- | :--- |
| `--dataset-path` | Path to validation dataset (images + .txt captions) |
| `--ckpt-path` | Path to the base SDXL checkpoint (single file) |
| `--lora-dir` | Directory containing checkpoints named `*-step[number].safetensors` |
| `--decay-type`, `-d` | LR decay schedule to use for merging: [1-sqrt, linear] (default: `1-sqrt`) |
| `--range`, `-r` | Merge a specified range of checkpoints |
| `--prediction-type`, `-p` | Use `epsilon` or `v_prediction` (default: `epsilon`) |
| `--min-snr-gamma` | (Optional) Apply Min-SNR-Gamma loss weighting |
| `--val-seed` | Random seed for validation |
| `--val-num-timesteps`, `-t` | Number of timesteps to use to calculate validation loss (default: 4) |
| `--device` | Compute device (default: `cuda`) |

## Search strategy

An exhaustive search requires evaluating $O(N^2)$ merge windows, which can take longer than training the LoRA itself. Instead, we take a greedy approach that assumes that the best individual checkpoint is inside the optimal merge window. We start from this checkpoint and expand the merge window outward until validation loss stops improving. This reduces the search complexity to $O(N)$.
