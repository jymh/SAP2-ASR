import json
import pathlib
import os
import argparse

HOME_DIR = pathlib.Path(os.path.expanduser("~"))

def make_filtered_train_dataset(prediction_file, raw_test_file, output_file, context_token=False):
    prediction_keywords_file = []
    with open(prediction_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            prediction_keywords_file.append(json.loads(line))
    with open(raw_test_file, 'r', encoding='utf8') as f:
        data = json.load(f)
        
    if context_token:
        INSTRUCTION_PROMPT_WITH_KEYWORDS = """Transcribe speech to text according to keywords may appear in the utterance. Possible keywords are: <|startofcontext|>{keywords}<|endofcontext|>"""
        INSTRUCTION_PROMPT_WITHOUT_KEYWORDS = """Transcribe speech to text.<|startofcontext|><|endofcontext|>"""
    else:
        INSTRUCTION_PROMPT_WITH_KEYWORDS = """Transcribe speech to text according to keywords may appear in the utterance. Possible keywords are: {keywords}"""
        INSTRUCTION_PROMPT_WITHOUT_KEYWORDS = """Transcribe speech to text."""
    # INSTRUCTION_PROMPT = """Select keywords that may appear in the speech from the following keywords list: <|startofcontext|>{keywords}<|endofcontext|>"""

    INPUT_WRAPPER = """<audio>{query}"""
    
    if "text" in data[0]:
        label_name = "text"
    elif "transcript" in data[0]:
        label_name = "transcript"
    elif "label" in data[0]:
        label_name = "label"
    else:
        raise ValueError("wrong label name")
    
    dataset_to_write = []
    predicted_data_idx = 0
    for item in data:
        audio_path = item["audio_path"].replace("/data/rym", str(HOME_DIR))
        if  audio_path != prediction_keywords_file[predicted_data_idx]['audios']:
            query = INSTRUCTION_PROMPT_WITHOUT_KEYWORDS
        elif prediction_keywords_file[predicted_data_idx]['response'] == "none":
            query = INSTRUCTION_PROMPT_WITHOUT_KEYWORDS
            assert audio_path == prediction_keywords_file[predicted_data_idx]['audios']
            predicted_data_idx += 1
        else:
            query = INSTRUCTION_PROMPT_WITH_KEYWORDS.format(keywords=prediction_keywords_file[predicted_data_idx]['response'])
            assert audio_path == prediction_keywords_file[predicted_data_idx]['audios']
            predicted_data_idx += 1 
        
        user_input = INPUT_WRAPPER.format(query=query)
        
        dataset_to_write.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content":  user_input,
                    },
                    {
                        "role": "assistant",
                        "content": item[label_name].lower()
                    }
                ],
                "audios": audio_path
            }
        )

    with open(output_file, 'w', encoding='utf8') as f:
        json.dump(dataset_to_write, f, indent=2, ensure_ascii=False)
        
# make_filtered_train_dataset(prediction_file="/home/rym/output/slidespeech_L95_lora_5slides_compress_filter/window_size_2_no_param/v0-20241230-123825/checkpoint-30026/test.jsonl",
#                             raw_test_file="data/context_filter/slidespeech_test_5slides_filter_keywords.json",
#                             output_file="data/add_context_token/slidespeech_L95_5slides_filtered_train/test_from_window_size2_no_param.json")
        
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--filtered_keywords_file",
        required=True,
    )
    parser.add_argument(
        "--raw_test_file",
        required=True,
    )
    parser.add_argument(
        "--output_file",
        required=True,
    )
    
    parser.add_argument(
        "--add_context_token",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    make_filtered_train_dataset(args.filtered_keywords_file, args.raw_test_file, args.output_file, args.add_context_token)

if __name__=="__main__":
    main()
    # make_filtered_train_dataset("/home/rym/output/slidespeech_L95_lora_organizedocr_filter_compress/window_size_2_no_param/v0-20250107-122257/checkpoint-28132/window_size2_no_param/test.jsonl",
    #                             "/home/rym/Projects/ms-swift/data/context_filter/slidespeech_test_organizedocr_filter_keywords.json",
    #                             "/home/rym/Projects/ms-swift/data/add_context_token/slidespeech_L95_organizedocr_filtered_train/test_from_window_size2_no_param.json")
        
    
    