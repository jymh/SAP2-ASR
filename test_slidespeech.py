import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import codecs
from pathlib import Path
import json

import subprocess
import pathlib
import re

def evaluate(ckpt_dir):
    INSTRUCTION_PROMPT = """请结合以下可能出现的关键词，做语音转文本。可能出现的关键词为：{keywords}"""

    INPUT_WRAPPER = """<audio>{audio_path}</audio>{query}"""


    MODEL_DIR = ckpt_dir
    save_dir = Path(MODEL_DIR) / "infer_result"
    save_dir.mkdir(parents=True, exist_ok=True)

    from swift.llm import (
        get_model_tokenizer, get_template, inference, ModelType,
        get_default_template_type, inference_stream
    )
    from swift.utils import seed_everything
    import torch

    model_type = ModelType.qwen2_audio_7b_instruct
    model_id_or_path = MODEL_DIR

    template_type = get_default_template_type(model_type)
    print(f'template_type: {template_type}')

    model, tokenizer = get_model_tokenizer(model_type, torch.float16, model_id_or_path=model_id_or_path,
                                        model_kwargs={'device_map': 'auto'})
    model.generation_config.max_new_tokens = 256
    template = get_template(template_type, tokenizer)
    seed_everything(42)

    with open("data/slidespeech_L95_all/test.json", 'r', encoding='utf8') as f:
        data = json.load(f)

    asr_result = []
    for item in data:
        keywords_lst = [s.lower() for s in item['keywords']]
        query = INSTRUCTION_PROMPT.format(keywords='\t'.join(keywords_lst))
        query = f'<audio>{query}'
        audios = [item['audio_path']]
        response, history = inference(model, template, query, audios=audios)
        print(f'query: {query}')
        print(f'response: {response}')
        asr_result.append(response)
        
    with codecs.open(str(save_dir / "test.ref"), mode='w', encoding='utf8') as f:
        for item in data:
            f.write(f'{item['audio_path']} {item['text']}' + '\n')

    with codecs.open(str(save_dir / "test.hyp"), mode='w', encoding='utf8') as f:
        for i, item in enumerate(asr_result):
            f.write(f'{data[i]['audio_path']} {item}' + '\n')


def convert_jsonl_to_txt(result_file):
    if isinstance(result_file, str):
        result_file = pathlib.Path(result_file)
    result_dir = result_file.parents[0]
        
    audio_paths = []
    hyps = []
    refs = []
    with result_file.open(mode='rt', encoding='utf8') as f:
        for line in f.readlines():
            data_item = json.loads(line)
            audio_paths.append(re.match(pattern=r"<audio>(.*?)</audio>", string=data_item["query"]).group(1))
            hyps.append(data_item["response"])
            refs.append(data_item["label"])
    
    hyp_writer =  codecs.open(str(result_dir / "test.hyp"), mode='w', encoding='utf8')
    ref_writer = codecs.open(str(result_dir / "test.ref"), mode='w', encoding='utf8')
    for i, item in enumerate(audio_paths):
        hyp_writer.write(f'{item} {hyps[i]}' + '\n')
        ref_writer.write(f'{item} {refs[i]}' + '\n')
        
    hyp_writer.close()
    ref_writer.close()
    
    
if __name__=="__main__":
    # result_file = pathlib.Path("/data/ymrong/output/qwen2-audio-7b-instruct/v11-20241011-105159/checkpoint-1450/infer_result/20241011-154337.jsonl")
    use_lora = True
    
    # model_path = pathlib.Path("/data/ymrong/output/slidespeech_L95_all_lora/qwen2-audio-7b-instruct/v0-20241011-234409/checkpoint-26754-merged")
    # eval_dataset = "data/slidespeech_L95_all/test.json"
    
    # model_path = pathlib.Path("/data/ymrong/output/slidespeech_30k_lora/qwen2-audio-7b-instruct/v0-20241013-180303/checkpoint-1347")
    # eval_dataset = "data/slidespeech_30k/test.json"
    
    # model_path = pathlib.Path("/data/ymrong/output/slidespeech_30k_lora/qwen2-audio-7b-instruct/v0-20241013-180303/checkpoint-1347")
    # eval_dataset = "data/slidespeech_30k/train.json"
    
    # model_path = pathlib.Path("/data/ymrong/output/qwen2-audio-7b-instruct/v11-20241011-105159/checkpoint-1450")
    # eval_dataset = "data/slidespeech_30k/test.json"
    
    # model_path = pathlib.Path("/data/ymrong/output/slidespeech_30k_full_freezevit/qwen2-audio-7b-instruct/v0-20241013-214952/checkpoint-1346")
    # eval_dataset = "data/slidespeech_30k/test.json"
    
    # model_path = pathlib.Path("/data/ymrong/output/slidespeech_30k_projector/qwen2-audio-7b-instruct/v2-20241014-140222/checkpoint-674")
    # eval_dataset = "data/slidespeech_30k/test.json"
    
    # model_path = pathlib.Path("/data/ymrong/output/slidespeech_30k_lora/qwen2-audio-7b/v0-20241017-183855/checkpoint-449")
    # eval_dataset = "data/slidespeech_30k_en_instruction/test.json"
    
    model_path = pathlib.Path("/data/ymrong/output/slidespeech_30k_lora_en_instruction/qwen2-audio-7b-instruct/v0-20241018-154420/checkpoint-449")
    eval_dataset = "data/slidespeech_30k_en_instruction/test.json"
    
    # model_path = pathlib.Path("/data/ymrong/output/slidespeech_L95_lora/qwen2-audio-7b/v1-20241017-215929/checkpoint-6689")
    # eval_dataset = "data/slidespeech_L95_en_instruction/test.json"
    
    # model_path = pathlib.Path("/data/ymrong/output/slidespeech_30k_filter_lora_en_instruction/qwen2-audio-7b-instruct/v6-20241021-112621/checkpoint-449")
    # eval_dataset = "data/slidespeech_30k_filter_en_instruction/test.json"
    
    # model_path = pathlib.Path("/data/ymrong/output/slidespeech_30k_train_filterkeywords_lora/qwen2-audio-7b-instruct/v0-20241023-181414/checkpoint-808")
    # eval_dataset = "data/slidespeech_30k_filtered_train/test.json"
    
    # 这个好像有问题，测试集处理错了
    # model_path = pathlib.Path("/data/ymrong/output/slidespeech_30k_train_filterkeywords_lora/qwen2-audio-7b-instruct/v1-20241023-201622/checkpoint-808")
    # eval_dataset = "data/slidespeech_30k_filtered_train/test.json"   
    
    # model_path = pathlib.Path("/data/ymrong/output/slidespeech_30k_train_filterkeywords_lora_en_instruction/qwen2-audio-7b-instruct/v2-20241024-110551/checkpoint-1616")
    # eval_dataset = "data/slidespeech_30k_filtered_train_en_instruction/test.json"
    
    # model_path = pathlib.Path("/data/ymrong/output/slidespeech_30k_train_filterkeywords_lora/qwen2-audio-7b-instruct/v4-20241024-134853/checkpoint-1616")
    # eval_dataset = "data/slidespeech_30k_filtered_train/test.json"
    
    if use_lora:
        original_ckpt_dir = model_path
        
        ckpt_dir = original_ckpt_dir.parents[0] / (str(original_ckpt_dir.name) + "-merged")
        if not ckpt_dir.exists():
            result = subprocess.run(["swift", "export", 
                        "--ckpt", str(original_ckpt_dir),
                        "--merge_lora", "true"])
            print(result.stdout)
        
    else:
        ckpt_dir = model_path
    
    # evaluate(ckpt_dir=ckpt_dir)
    
    subprocess.run(["swift", "infer", "--ckpt_dir", str(ckpt_dir), "--val_dataset", eval_dataset])
    
    result_file = next((Path(ckpt_dir) / "infer_result").iterdir())
    convert_jsonl_to_txt(result_file)

    hyp_file = result_file.parents[0] / "test.hyp"
    ref_file = result_file.parents[0] / "test.ref"
    wer_file = result_file.parents[0] / "test.wer"
    import subprocess

    # result = subprocess.run(["python", "/data/ymrong/Projects/wenet/tools/compute-wer.py", str(ref_file), str(hyp_file), ">", str(wer_file)], shell=True, capture_output=True, text=True)
    result = subprocess.run(f"python /data/ymrong/Projects/wenet/tools/compute-wer.py {str(ref_file)} {str(hyp_file)} > {str(wer_file)}",
                            shell=True,
                            capture_output=True,
                            text=True)
    print(result.stdout)



        

# # 流式（streaming）
# # query = '这段语音是男生还是女生'
# # gen = inference_stream(model, template, query, history, audios=audios)
# # print_idx = 0
# # print(f'query: {query}\nresponse: ', end='')
# # for response, history in gen:
# #     delta = response[print_idx:]
# #     print(delta, end='', flush=True)
# #     print_idx = len(response)
# # print()
# # print(f'history: {history}')
"""
query: <audio>这段语音说了什么
response: 这段语音说的是:'今天天气真好呀'
query: 这段语音是男生还是女生
response: 男声。
history: [['<audio>这段语音说了什么', "这段语音说的是:'今天天气真好呀'"], ['这段语音是男生还是女生', '男声。']]
"""