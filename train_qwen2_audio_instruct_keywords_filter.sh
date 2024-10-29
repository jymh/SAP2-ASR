# # 30k lora
# NPROC_PER_NODE=6 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 swift sft \
#     --model_id_or_path "/data/ymrong/models/qwen2-audio-instruct"  \
#     --model_type qwen2-audio-7b-instruct \
#     --dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k_filter_en_instruction/train.json" \
#     --val_dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_30k_filter_en_instruction/dev.json" \
#     --num_train_epochs 1 \
#     --batch_size 12 \
#     --output_dir "/data/ymrong/output/slidespeech_30k_filter_lora_en_instruction" \
#     --sft_type lora \
#     --deepspeed default-zero2


# # L95 lora
NPROC_PER_NODE=3 CUDA_VISIBLE_DEVICES=2,3,4 swift sft \
    --model_id_or_path "/data/ymrong/models/qwen2-audio-instruct"  \
    --model_type qwen2-audio-7b-instruct \
    --dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_L95_filter_en_instruction/train.json" \
    --val_dataset "/data/ymrong/Projects/ms-swift/data/slidespeech_L95_filter_en_instruction/dev.json" \
    --num_train_epochs 1 \
    --batch_size 4 \
    --output_dir "/data/ymrong/output/slidespeech_L95_filter_lora_en_instruction" \
    --sft_type lora \
    --deepspeed default-zero2