#!/bin/bash

# L95 lora 5 slides multitask
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model_id_or_path "/data8/rym/models/qwen2-audio-instruct"  \
    --model_type qgc-qwen2-audio-7b-instruct \
    --dataset "/data8/rym/Projects/ms-swift/data/slidespeech_L95_5slidesocr_en_instruction/train.json" \
    --val_dataset "/data8/rym/Projects/ms-swift/data/slidespeech_L95_5slidesocr_en_instruction/dev.json" \
    --dataloader_num_workers 0 \
    --eval_steps 2000 \
    --num_train_epochs 1 \
    --batch_size 3 \
    --max_length 4096 \
    --output_dir "/data8/rym/output/slidespeech_L95_lora_5slidesocr_en_instruction" \
    --sft_type qgcpeft \
    --lora_rank 32 \
    --qgc_window_size 8 \
    --compressor_hidden_size 4096 \
    --num_attention_heads 4 \
    --deepspeed default-zero2