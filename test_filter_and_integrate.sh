#! /bin/bash

PROJECT_DIR=${HOME}/Projects/ms-swift
OUTPUT_DIR=/data8/rym/output


# filter_adapter_dir=${OUTPUT_DIR}/slidespeech_L95_lora_organizedocr_filter_compress/window_size_2_no_param/v0-20250107-122257/checkpoint-28132
# asr_adapter_dir=${OUTPUT_DIR}/slidespeech_L95_lora_organizedocr_filtered_train_compress/window_size_2_no_param/v1-20250108-034742/checkpoint-30099
# result_dir="window_size2_no_param"
# filter_eval_dataset=${PROJECT_DIR}/data/add_context_token/slidespeech_L95_organizedocr_filter/test.json
# raw_test_file=${PROJECT_DIR}/data/context_filter/slidespeech_test_organizedocr_filter_keywords.json
# asr_eval_dataset=${PROJECT_DIR}/data/add_context_token/slidespeech_L95_organizedocr_filtered_train/test_from_window_size2_no_param.json

# filter_adapter_dir=${OUTPUT_DIR}/slidespeech_L95_lora_5slides_organizedocr_filter_compress/window_size_2_no_param/v0-20250108-102705/checkpoint-34359
# asr_adapter_dir=${OUTPUT_DIR}/slidespeech_L95_lora_5slides_organizedocr_filtered_train_compress/window_size_2_no_param/v0-20250109-072632/checkpoint-34399
# result_dir="window_size2_no_param"
# filter_eval_dataset=${PROJECT_DIR}/data/add_context_token/slidespeech_L95_5slides_organizedocr_filter/test.json
# raw_test_file=${PROJECT_DIR}/data/context_filter/slidespeech_test_5slides_organizedocr_filter_keywords.json
# asr_eval_dataset=${PROJECT_DIR}/data/add_context_token/slidespeech_L95_5slides_organizedocr_filtered_train/test_from_window_size2_no_param.json

filter_adapter_dir=${OUTPUT_DIR}/slidespeech_L95_lora_organizedocr_filter_compress/window_size_2_no_param/v1-20250111-004915/checkpoint-37509
asr_adapter_dir=${OUTPUT_DIR}/slidespeech_L95_lora_organizedocr_filtered_train_compress/window_size_2_no_param/v0-20250111-125757/checkpoint-40132
result_dir="window_size2_no_param"
filter_eval_dataset=${PROJECT_DIR}/data/add_context_token/slidespeech_L95_organizedocr_filter/test.json
raw_test_file=${PROJECT_DIR}/data/context_filter/slidespeech_test_organizedocr_filter_keywords.json
asr_eval_dataset=${PROJECT_DIR}/data/add_context_token/slidespeech_L95_organizedocr_filtered_train/test_from_window_size2_no_param.json


qgc_window_size=2

mkdir ${filter_adapter_dir}/${result_dir}
result_file=${filter_adapter_dir}/${result_dir}/test.jsonl

# CUDA_VISIBLE_DEVICES=7 swift infer \
#     --model /home/rym/models/qwen2-audio-instruct \
#     --adapters ${filter_adapter_dir} \
#     --infer_backend pt \
#     --temperature 0 \
#     --val_dataset ${filter_eval_dataset} \
#     --result_path ${result_file} \
#     --stream true \
#     --qgc_window_size ${qgc_window_size} \
#     --compressor_hidden_size 4096 \
#     --num_attention_heads 4

python extract_predicted_keywords.py \
    --filtered_keywords_file ${result_file} \
    --raw_test_file ${raw_test_file} \
    --output_file ${asr_eval_dataset}

mkdir ${asr_adapter_dir}/${result_dir}
result_file=${asr_adapter_dir}/${result_dir}/test.jsonl

# CUDA_VISIBLE_DEVICES=7 swift infer \
#     --model /home/rym/models/qwen2-audio-instruct \
#     --adapters ${asr_adapter_dir} \
#     --infer_backend pt \
#     --temperature 0 \
#     --val_dataset ${asr_eval_dataset} \
#     --result_path ${result_file} \
#     --stream false \
#     --qgc_window_size ${qgc_window_size} \
#     --compressor_hidden_size 4096 \
#     --num_attention_heads 4


python evaluate_slidespeech_process.py --input_file ${result_file}

superdir=${asr_adapter_dir}/${result_dir}
python $HOME/Projects/SLAM-LLM/src/slam_llm/utils/whisper_tn.py ${superdir}/test.ref ${superdir}/test.ref.proc
python $HOME/Projects/SLAM-LLM/src/slam_llm/utils/whisper_tn.py ${superdir}/test.hyp ${superdir}/test.hyp.proc

python compute_wer_details.py --v 1 \
    --ref ${superdir}/test.ref.proc \
    --ref_ocr data/test.ocr  \
    --ref2session data/test.wav2session \
    --rec_name base \
    --rec_file ${superdir}/test.hyp.proc \
    > ${superdir}/test.proc.werall


