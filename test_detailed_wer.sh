
# superdir="/data/ymrong/output/slidespeech_30k_lora/qwen2-audio-7b-instruct/v0-20241013-180303/checkpoint-1347-merged/infer_result"
# superdir="/data/ymrong/output/slidespeech_30k_projector/qwen2-audio-7b-instruct/v2-20241014-140222/checkpoint-674/infer_result"
# superdir="/data/ymrong/output/slidespeech_L95_all_lora/qwen2-audio-7b-instruct/v0-20241011-234409/checkpoint-26754-merged/infer_result"
# superdir="/data/ymrong/output/slidespeech_30k_lora_multitask_train_en_instruction/qwen2-audio-7b-instruct/v3-20241028-162459/checkpoint-1346-merged/infer_result"
superdir="/data/ymrong/output/slidespeech_30k_train_filterkeywords_lora_en_instruction/qwen2-audio-7b-instruct/v2-20241024-110551/checkpoint-1616-merged/infer_result_from_filter"


python compute_wer_details.py --v 1 \
    --ref ${superdir}/test.ref \
    --ref_ocr data/test.ocr  \
    --ref2session data/test.wav2session \
    --rec_name base \
    --rec_file ${superdir}/test.hyp \
    > ${superdir}/test.werall