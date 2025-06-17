from minicons import scorer
import torch
import numpy as np
import csv
import os

def load_sentences(filepath):
    sentence_pairs = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        for row in reader:
            good_sentence = row[0]
            bad_sentence = row[1]
            sentence_pairs.append([good_sentence, bad_sentence])
    return sentence_pairs

def compute_score(data, model, mode):
    if mode == 'ilm':
        score = model.sequence_score(data, reduction=lambda x: x.sum(0).item())
    elif mode == 'mlm':
        score = model.sequence_score(data, reduction=lambda x: x.sum(0).item(), PLL_metric='within_word_l2r')
    return score

def process_files(model, mode, model_name, output_folder):
    file_names = ["argument_structure_ditransitive_original.csv",
              "argument_structure_ditransitive_ken.csv",
              "argument_structure_ditransitive_dik.csv",
              "argument_structure_ditransitive_inca.csv",
              "argument_structure_ditransitive_OSV.csv",
              "argument_structure_ditransitive_OVS.csv",
              "argument_structure_ditransitive_finite.csv",
              "argument_structure_ditransitive_SOV.csv",
              "argument_structure_ditransitive_SVO.csv",
              "argument_structure_ditransitive_VOS.csv",
              "argument_structure_ditransitive_VSO.csv",
              "argument_structure_transitive_original.csv",
              "argument_structure_transitive_ken.csv",
              "argument_structure_transitive_dik.csv",
              "argument_structure_transitive_inca.csv",
              "argument_structure_transitive_OSV.csv",
              "argument_structure_transitive_OVS.csv",
              "argument_structure_transitive_finite.csv",
              "argument_structure_transitive_SOV.csv",
              "argument_structure_transitive_SVO.csv",
              "argument_structure_transitive_VOS.csv",
              "argument_structure_transitive_VSO.csv"]

    os.makedirs(output_folder, exist_ok=True)

    for file_path in file_names:
        try:
            pairs = load_sentences(file_path)
            results = []
            differences = 0
            accuracy = 0

            for pair in pairs:
                score = compute_score(pair, model, mode)
                results.append({
                    'good_sentence': pair[0],
                    'bad_sentence': pair[1],
                    'good_score': score[0],
                    'bad_score': score[1],
                    'difference': score[0] - score[1],
                    'correct': score[0] > score[1]
                })

                if score[0] > score[1]:
                    accuracy += 1
                differences += score[0] - score[1]

            mean_difference = differences / len(pairs)
            accuracy = accuracy / len(pairs)

            summary = {
                'file_name': file_path,
                'mean_difference': mean_difference,
                'accuracy': accuracy,
                'total_pairs': len(pairs)
            }

            output_file = os.path.join(output_folder, f"{model_name}_{file_path}")
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

            print(f"Processed {file_path}:")
            print(f"  Mean difference: {mean_difference:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

ilm_model_names = ['meta-llama/Llama-3.1-8B',
                   'google/gemma-2-9b',
                   'CohereForAI/aya-expanse-8b',
                   'Qwen/Qwen2.5-7B',
                   'utter-project/EuroLLM-9B',
                   'ytu-ce-cosmos/turkish-gpt2-large',
                   'goldfish-models/tur_latn_1000mb',
                   'goldfish-models/tur_latn_100mb',
                   'goldfish-models/tur_latn_10mb',
                   'goldfish-models/tur_latn_5mb',
                   'google/gemma-3-4b-pt',
                   'google/gemma-3-12b-pt']

mlm_model_names = ['dbmdz/bert-base-turkish-128k-uncased']


device = 'cuda' if torch.cuda.is_available() else 'cpu'

mode = 'mlm'
model_name = mlm_model_names[0]
model = scorer.MaskedLMScorer(model_name, device)

process_files(
    model=model,
    mode=mode,
    model_name=model_name,
    output_folder='/scores'
)