#!/bin/bash

# L95 lora 5 slides multitask
# NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
#     --model_id_or_path "/data/rym/models/qwen2-audio-instruct"  \
#     --model_type sap-qwen2-audio-7b-instruct \
#     --dataset "/data/rym/Projects/ms-swift/data/slidespeech_L95_5slidesocr_en_instruction/train.json" \
#     --val_dataset "/data/rym/Projects/ms-swift/data/slidespeech_L95_5slidesocr_en_instruction/dev.json" \
#     --dataloader_num_workers 0 \
#     --eval_steps 2000 \
#     --num_train_epochs 1 \
#     --batch_size 3 \
#     --max_length 4096 \
#     --output_dir "/data/rym/output/slidespeech_L95_lora_5slidesocr_en_instruction" \
#     --sft_type sappeft \
#     --lora_rank 32 \
#     --sap_window_size 8 \
#     --compressor_hidden_size 4096 \
#     --num_attention_heads 4 \
#     --deepspeed default-zero2


# NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
#     --model "/data/rym/models/sap-qwen2-audio-instruct"  \
#     --model_type sap_qwen2_audio \
#     --dataset "/data/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_5slides_multitask_train_en_instruction/train.json" \
#     --val_dataset "/data/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_5slides_multitask_train_en_instruction/dev.json" \
#     --save_steps 2000 \
#     --num_train_epochs 1 \
#     --save_total_limit 2 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --max_length 4096 \
#     --output_dir "/data/rym/output/slidespeech_L95_lora_5slides_multitask_train_compress_en_instruction/window_size_4" \
#     --train_type sappeft \
#     --freeze_llm false \
#     --freeze_vit true \
#     --freeze_aligner false \
#     --lora_rank 8 \
#     --sap_window_size 4 \
#     --compressor_hidden_size 4096 \
#     --num_attention_heads 4 \
#     --deepspeed zero2

#     --gradient_checkpointing_kwargs "{\"use_reentrant\": false}" \

# NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
#     --model "/data/rym/models/qwen2-audio-instruct"  \
#     --model_type sap_qwen2_audio \
#     --dataset "/data/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_5slides_multitask_train_en_instruction/train.json" \
#     --val_dataset "/data/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_5slides_multitask_train_en_instruction/dev.json" \
#     --save_steps 2000 \
#     --num_train_epochs 1 \
#     --save_total_limit 2 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --max_length 4096 \
#     --output_dir "/data/rym/output/slidespeech_L95_lora_5slides_multitask_train_compress_en_instruction/window_size_2_simplify" \
#     --train_type sappeft \
#     --freeze_llm false \
#     --freeze_vit true \
#     --freeze_aligner false \
#     --lora_rank 8 \
#     --sap_window_size 2 \
#     --compressor_hidden_size 4096 \
#     --num_attention_heads 4 \
#     --deepspeed zero2

# NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=4,5,6,7 swift sft \
#     --model "/data/rym/models/qwen2-audio-instruct"  \
#     --model_type sap_qwen2_audio \
#     --dataset "/data/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_filter/train.json" \
#     --val_dataset "/data/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_filter/dev.json" \
#     --save_steps 2000 \
#     --save_total_limit 2 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --max_length 4096 \
#     --output_dir "/data/rym/output/slidespeech_L95_lora_filter_compress/window_size_2_no_param" \
#     --train_type lora \
#     --freeze_llm false \
#     --freeze_vit true \
#     --freeze_aligner false \
#     --lora_rank 8 \
#     --sap_window_size 2 \
#     --compressor_hidden_size 4096 \
#     --num_attention_heads 4 \
#     --deepspeed zero2


# NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=4,5,6,7 swift sft \
#     --model "/data/rym/models/qwen2-audio-instruct"  \
#     --model_type sap_qwen2_audio \
#     --dataset "/data/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_filtered_train/train.json" \
#     --val_dataset "/data/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_filtered_train/dev.json" \
#     --save_steps 2000 \
#     --save_total_limit 2 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --max_length 4096 \
#     --output_dir "/data/rym/output/slidespeech_L95_lora_filtered_train_compress/window_size_2_no_param" \
#     --train_type lora \
#     --freeze_llm false \
#     --freeze_vit true \
#     --freeze_aligner false \
#     --lora_rank 8 \
#     --sap_window_size 2 \
#     --compressor_hidden_size 4096 \
#     --num_attention_heads 4 \
#     --deepspeed zero2

# NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
#     --model "/data/rym/models/qwen2-audio-instruct"  \
#     --model_type sap_qwen2_audio \
#     --dataset "/data/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_multitask/train.json" \
#     --val_dataset "/data/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_multitask/dev.json" \
#     --save_steps 1000 \
#     --save_total_limit 2 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --max_length 4096 \
#     --output_dir "/data/rym/output/slidespeech_L95_lora_multitask_compress/window_size_2_no_param" \
#     --train_type lora \
#     --freeze_llm false \
#     --freeze_vit true \
#     --freeze_aligner false \
#     --lora_rank 8 \
#     --sap_window_size 2 \
#     --compressor_hidden_size 4096 \
#     --num_attention_heads 4 \
#     --deepspeed zero2

NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=1,6 swift sft \
    --model "/data/rym/models/qwen2-audio-instruct"  \
    --model_type sap_qwen2_audio \
    --dataset "/data/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_en_instruction/train.json" \
    --val_dataset "/data/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_en_instruction/dev.json" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --max_length 4096 \
    --output_dir "/data/rym/output/slidespeech_L95_lora_compress/test_code" \
    --train_type lora \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner false \
    --lora_rank 8 \
    --sap_window_size 2 \
    --compressor_hidden_size 4096 \
    --num_attention_heads 4 \
    --deepspeed zero2