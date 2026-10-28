#!/bin/bash

# Continual training of Luna-TTS Realtime from an OmniVoice (bidirectional
# masked-diffusion) checkpoint. Tokenization is the same as a normal fine-tune;
# only the objective and attention mask change.

set -euo pipefail

GPU_IDS="${GPU_IDS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_config_luna_realtime.json}"
DATA_CONFIG="${DATA_CONFIG:-config/data_config_finetune.json}"
OUTPUT_DIR="${OUTPUT_DIR:-exp/luna_tts_realtime}"

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"

accelerate launch \
    --gpu_ids "${GPU_IDS}" \
    --num_processes "${NUM_GPUS}" \
    -m omnivoice.cli.train \
    --train_config "${TRAIN_CONFIG}" \
    --data_config "${DATA_CONFIG}" \
    --output_dir "${OUTPUT_DIR}"
