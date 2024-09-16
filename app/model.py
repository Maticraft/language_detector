from typing import List, Optional

import fasttext
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def download_model() -> fasttext.FastText._FastText:
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")

    return fasttext.load_model(model_path)

model = download_model()

def detect_language(text: str) -> str:
    prediction = model.predict(text)
    full_language = prediction[0][0].replace("__label__", "")
    lang_symbol = full_language.split("_")[0]
    return lang_symbol


def run_benchmark_test(languages_subset: Optional[List[str]] = None) -> float:
    input_texts_path = './samples/x_test.txt'
    output_labels_path = './samples/y_test.txt'
    with open(input_texts_path, 'r') as f:
        input_texts = f.readlines()
    with open(output_labels_path, 'r') as f:
        output_labels = f.readlines()

    correct_predictions = 0
    total_predictions = 0
    for text, label in tqdm(zip(input_texts, output_labels), desc="Running benchmark test"):
        label_parsed = label.strip().split("-")[0]
        should_evaluate = (not languages_subset) or (label_parsed in languages_subset)
        if should_evaluate:
            correct_predictions += int(is_prediction_correct(text, label_parsed))
            total_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

def is_prediction_correct(text: str, label: str) -> bool:
    predicted_language = detect_language(text.strip())
    min_num_chars = min(len(predicted_language), len(label))
    is_prediction_correct = predicted_language[:min_num_chars] == label[:min_num_chars]
    return is_prediction_correct
