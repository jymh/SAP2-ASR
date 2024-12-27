import os
import json
import subprocess
import pathlib
import re
import codecs
from pathlib import Path
import argparse

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
            audio_paths.append(data_item['audios'])
            hyps.append(data_item["response"])
            refs.append(data_item["labels"])
    
    hyp_writer =  codecs.open(str(result_dir / "test.hyp"), mode='w', encoding='utf8')
    ref_writer = codecs.open(str(result_dir / "test.ref"), mode='w', encoding='utf8')
    for i, item in enumerate(audio_paths):
        hyp_writer.write(f'{item} {hyps[i]}' + '\n')
        ref_writer.write(f'{item} {refs[i]}' + '\n')
        
    hyp_writer.close()
    ref_writer.close()
    
def convert_jsonl_to_txt_multitask(result_file):
    if isinstance(result_file, str):
        result_file = pathlib.Path(result_file)
    result_dir = result_file.parents[0]
        
    audio_paths = []
    hyps = []
    refs = []
    with result_file.open(mode='rt', encoding='utf8') as f:
        for line in f.readlines():
            data_item = json.loads(line)
            if re.search(pattern=r'Transcription: (.+)', string=data_item["labels"]) is not None and re.search(pattern=r'Transcription: (.+)', string=data_item["response"]) is not None:
                audio_paths.append(data_item['audios'])
                refs.append(re.search(pattern=r'Transcription: (.+)', string=data_item["labels"]).group(1))
                hyps.append(re.search(pattern=r'Transcription: (.+)', string=data_item["response"]).group(1))   
    
    hyp_writer =  codecs.open(str(result_dir / "test.hyp"), mode='w', encoding='utf8')
    ref_writer = codecs.open(str(result_dir / "test.ref"), mode='w', encoding='utf8')
    for i, item in enumerate(audio_paths):
        hyp_writer.write(f'{item} {hyps[i]}' + '\n')
        ref_writer.write(f'{item} {refs[i]}' + '\n')
        
    hyp_writer.close()
    ref_writer.close()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_file",
        help="File path to the inference results of ASR model",
        required=True,
    )
    parser.add_argument(
        "--multitask",
        help="Whether the ASR result is infered with a multitask format",
        action="store_true",
        default=False,
    )
    
    args = parser.parse_args()
    
    if args.multitask:
        convert_jsonl_to_txt_multitask(args.input_file)
    else:
        convert_jsonl_to_txt(args.input_file)

if __name__=="__main__":
    main()
    
