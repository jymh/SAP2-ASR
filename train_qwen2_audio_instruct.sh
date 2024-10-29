#!/bin/bash

# # 30k full freezevit

# NPROC_PER_NODE=6 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 swift sft \
#     --model_id_or_path "/data/ymrong/models/qwen2-audio-instruct"  \
#     --model_type qwen2-audio-7b-instruct \
#     --dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k/train.json" \
#     --val_dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k/dev.json" \
#     --num_train_epochs 1 \
#     --batch_size 2 \
#     --output_dir "/data/ymrong/output/slidespeech_30k_full_freezevit" \
#     --sft_type full \
#     --deepspeed default-zero2 \
#     --freeze_vit True

# NPROC_PER_NODE=6 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 swift sft \
#     --model_id_or_path "/data/ymrong/models/qwen2-audio-instruct"  \
#     --model_type qwen2-audio-7b-instruct \
#     --dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k/train.json" \
#     --val_dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k/dev.json" \
#     --num_train_epochs 1 \
#     --batch_size 2 \
#     --output_dir "/data/ymrong/output/slidespeech_30k_full_freezevit_test" \
#     --sft_type full \
#     --freeze_parameters 1 \
#     --additional_trainable_parameters language_model multi_modal_projector \
#     --deepspeed default-zero2

# # 30k only train projector

# NPROC_PER_NODE=6 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 swift sft \
#     --model_id_or_path "/data/ymrong/models/qwen2-audio-instruct"  \
#     --model_type qwen2-audio-7b-instruct \
#     --dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k/train.json" \
#     --val_dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k/dev.json" \
#     --num_train_epochs 1 \
#     --batch_size 12 \
#     --output_dir "/data/ymrong/output/slidespeech_30k_projector" \
#     --sft_type full \
#     --freeze_parameters_ratio 1 \
#     --additional_trainable_parameters multi_modal_projector \
#     --deepspeed default-zero2


# # 30k lora
# NPROC_PER_NODE=5 CUDA_VISIBLE_DEVICES=0,1,2,3,4 swift sft \
#     --model_id_or_path "/data/ymrong/models/qwen2-audio-instruct"  \
#     --model_type qwen2-audio-7b-instruct \
#     --dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k_en_instruction/train.json" \
#     --val_dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k_en_instruction/dev.json" \
#     --num_train_epochs 1 \
#     --batch_size 12 \
#     --output_dir "/data/ymrong/output/slidespeech_30k_lora_en_instruction" \
#     --sft_type lora \
#     --deepspeed default-zero2


# # 30k lora filtered keywords
# NPROC_PER_NODE=5 CUDA_VISIBLE_DEVICES=0,1,2,3,4 swift sft \
#     --model_id_or_path "/data/ymrong/models/qwen2-audio-instruct"  \
#     --model_type qwen2-audio-7b-instruct \
#     --dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k_filtered_train/train.json" \
#     --val_dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k_filtered_train/dev.json" \
#     --num_train_epochs 1 \
#     --batch_size 4 \
#     --output_dir "/data/ymrong/output/slidespeech_30k_train_filterkeywords_lora" \
#     --sft_type lora \
#     --deepspeed default-zero2

# # 30k lora multitask
NPROC_PER_NODE=3 CUDA_VISIBLE_DEVICES=2,3,4 swift sft \
    --model_id_or_path "/data/ymrong/models/qwen2-audio-instruct"  \
    --model_type qwen2-audio-7b-instruct \
    --dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k_multitask_train_en_instruction/train.json" \
    --val_dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k_multitask_train_en_instruction/dev.json" \
    --eval_steps 500 \
    --num_train_epochs 1 \
    --batch_size 4 \
    --output_dir "/data/ymrong/output/slidespeech_30k_lora_multitask_train_en_instruction" \
    --sft_type lora \
    --deepspeed default-zero2
