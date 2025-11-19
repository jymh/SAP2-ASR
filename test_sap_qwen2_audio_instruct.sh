#! /bin/bash

PROJECT_DIR=${HOME}/Projects/ms-swift
OUTPUT_DIR=/data/rym/output


adapter_dir=${OUTPUT_DIR}/slidespeech_L95_lora_5slides_multitask_train_compress_en_instruction/v0-20241226-191733/checkpoint-30099
# result_dir="infer_result"
eval_dataset=${PROJECT_DIR}/data/add_context_token/slidespeech_L95_5slides_multitask_train_en_instruction/test.json

result_file=${adapter_dir}/test.jsonl

# CUDA_VISIBLE_DEVICES=0 swift infer \
#     --adapters ${adapter_dir} \
#     --infer_backend pt \
#     --temperature 0 \
#     --max_batch_size 4 \
#     --val_dataset ${eval_dataset} \
#     --result_path ${result_file} \
#     --stream false \
#     --sap_window_size 2 \
#     --compressor_hidden_size 4096 \
#     --num_attention_heads 4

python evaluate_slidespeech_process.py --input_file ${result_file} --multitask

superdir=${adapter_dir}/${result_dir}
python $HOME/Projects/SLAM-LLM/src/slam_llm/utils/whisper_tn.py ${superdir}/test.ref ${superdir}/test.ref.proc
python $HOME/Projects/SLAM-LLM/src/slam_llm/utils/whisper_tn.py ${superdir}/test.hyp ${superdir}/test.hyp.proc

python compute_wer_details.py --v 1 \
    --ref ${superdir}/test.ref.proc \
    --ref_ocr data/test.ocr  \
    --ref2session data/test.wav2session \
    --rec_name base \
    --rec_file ${superdir}/test.hyp.proc \
    > ${superdir}/test.proc.werall


