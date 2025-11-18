
# superdir="/data/ymrong/output/slidespeech_30k_lora/qwen2-audio-7b-instruct/v0-20241013-180303/checkpoint-1347-merged/infer_result"
# superdir="/data/ymrong/output/slidespeech_30k_projector/qwen2-audio-7b-instruct/v2-20241014-140222/checkpoint-674/infer_result"
# superdir="/data/ymrong/output/slidespeech_L95_all_lora/qwen2-audio-7b-instruct/v0-20241011-234409/checkpoint-26754-merged/infer_result"
# superdir="/data/ymrong/output/slidespeech_30k_lora_multitask_train_en_instruction/qwen2-audio-7b-instruct/v3-20241028-162459/checkpoint-1346-merged/infer_result"
# superdir="/data/ymrong/output/slidespeech_30k_train_filterkeywords_lora_en_instruction/qwen2-audio-7b-instruct/v2-20241024-110551/checkpoint-1616-merged/infer_result_from_filter"
# superdir="/data/rym/output/slidespeech_L95_train_filteredkeywords_lora_en_instruction/qwen2-audio-7b-instruct/v1-20241031-164443/checkpoint-1000-merged/infer_result"
# superdir="/data/rym/output/slidespeech_L95_train_filteredkeywords_lora_en_instruction/qwen2-audio-7b-instruct/v3-20241101-001300/checkpoint-20066-merged/infer_result_from_label"
# superdir="/data/rym/output/slidespeech_L95_train_filteredkeywords_lora_en_instruction/qwen2-audio-7b-instruct/v3-20241101-001300/checkpoint-20066-merged/infer_result"

# superdir=["/data/ymrong/output/slidespeech_30k_lora/qwen2-audio-7b-instruct/v0-20241013-180303/checkpoint-1347-merged/infer_result",
# "/data/ymrong/output/slidespeech_30k_projector/qwen2-audio-7b-instruct/v2-20241014-140222/checkpoint-674/infer_result",
# "/data/ymrong/output/slidespeech_L95_all_lora/qwen2-audio-7b-instruct/v0-20241011-234409/checkpoint-26754-merged/infer_result",
# "/data/ymrong/output/slidespeech_30k_lora_multitask_train_en_instruction/qwen2-audio-7b-instruct/v3-20241028-162459/checkpoint-1346-merged/infer_result",
# "/data/ymrong/output/slidespeech_30k_train_filterkeywords_lora_en_instruction/qwen2-audio-7b-instruct/v2-20241024-110551/checkpoint-1616-merged/infer_result_from_filter",
# "/data/rym/output/slidespeech_L95_train_filteredkeywords_lora_en_instruction/qwen2-audio-7b-instruct/v1-20241031-164443/checkpoint-1000-merged/infer_result",
# "/data/rym/output/slidespeech_L95_train_filteredkeywords_lora_en_instruction/qwen2-audio-7b-instruct/v3-20241101-001300/checkpoint-20066-merged/infer_result_from_label",
# "/data/rym/output/slidespeech_L95_train_filteredkeywords_lora_en_instruction/qwen2-audio-7b-instruct/v3-20241101-001300/checkpoint-20066-merged/infer_result"]

# superdir="/data/rym/output/slidespeech_L95_lora_5slidesocr_en_instruction/qwen2-audio-7b-instruct/v2-20241115-002608/checkpoint-30100-merged/infer_result"
# superdir="/data/rym/output/slidespeech_L95_lora_5slides_filter_en_instruction/qwen2-audio-7b-instruct/v0-20241117-001758/checkpoint-12000-merged/infer_result"
# superdir="/data/rym/output/slidespeech_L95_lora_5slides_filtered_train_en_instruction/qwen2-audio-7b-instruct/v0-20241119-031651/checkpoint-7525-merged/infer_result/"
# superdir="/data/rym/output/slidespeech_L95_lora_3slidesocr_en_instruction/qwen2-audio-7b-instruct/v2-20241201-021657/checkpoint-20066-merged/infer_result/"
# superdir="/data/rym/output/slidespeech_L95_lora_organizedocr/qwen2-audio-7b-instruct/v0-20241208-193552/checkpoint-14899-merged/infer_result/"
# superdir="/data/rym/output/slidespeech_L95_lora_raw/qwen2-audio-7b-instruct/v0-20241210-084331/checkpoint-15050-merged/infer_result/"
# superdir="/data/rym/output/slidespeech_L95_lora_5slidesocr_en_instruction/qgc-qwen2-audio-7b-instruct/v2-20241211-004321/checkpoint-30098-merged/infer_result/"
# superdir="/data/rym/output/slidespeech_L95_lora_5slides_multitask_train_en_instruction/qgc-qwen2-audio-7b-instruct/v1-20241211-170945/checkpoint-30099-merged/infer_result/"
# superdir="/data/rym/output/slidespeech_L95_lora_5slides_multitask_train_en_instruction/qgc-qwen2-audio-7b-instruct/v2-20241212-120605/checkpoint-22000-merged/infer_result/"
# superdir="/data/rym/output/slidespeech_L95_lora_5slidesocr_en_instruction/qgc-qwen2-audio-7b-instruct/v0-20241214-114047/checkpoint-30099-merged/infer_result/"
# superdir="/data/rym/output/slidespeech_L95_lora_5slides_multitask_train_en_instruction/qgc-qwen2-audio-7b-instruct/v4-20241216-113603/checkpoint-15049-merged/infer_result/
# superdir="/data/rym/output/slidespeech_L95_lora_5slidesocr_en_instruction/qgc-qwen2-audio-7b-instruct/v1-20241216-143511/checkpoint-30099-merged/infer_result/"
superdir="/data/rym/output/slidespeech_L95_lora_5slides_multitask_train_en_instruction/qgc-qwen2-audio-7b-instruct/v5-20241216-214322/checkpoint-30098-merged/infer_result/"

python $HOME/Projects/SLAM-LLM/src/slam_llm/utils/whisper_tn.py ${superdir}/test.ref ${superdir}/test.ref.proc
python $HOME/Projects/SLAM-LLM/src/slam_llm/utils/whisper_tn.py ${superdir}/test.hyp ${superdir}/test.hyp.proc


python compute_wer_details.py --v 1 \
    --ref ${superdir}/test.ref.proc \
    --ref_ocr data/test.ocr  \
    --ref2session data/test.wav2session \
    --rec_name base \
    --rec_file ${superdir}/test.hyp.proc \
    > ${superdir}/test.proc.werall