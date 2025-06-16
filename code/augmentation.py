import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import random
import jpype
import jpype.imports
from jpype.types import *
from functools import lru_cache
from transformers import pipeline
import torch
from collections import defaultdict
import traceback

ZEMBEREK_PATH = 'zemberek-full.jar'
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[''zemberek-full.jar''], convertStrings=True)

TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
TurkishSpellChecker = JClass('zemberek.normalization.TurkishSpellChecker')
morphology = TurkishMorphology.createWithDefaults()
spell_checker = TurkishSpellChecker(morphology)

device = 0 if torch.cuda.is_available() else -1
model_name = "dbmdz/bert-base-turkish-128k-uncased"
fill_mask = pipeline("fill-mask", model=model_name, tokenizer=model_name, device=device)

DISCOURSE_MARKERS = ["Yani", "Görünüşe göre", "Demek ki", "Anlaşılan",
                    "Öyle görünüyor ki", "Belli ki", "Anlaşılıyor ki",
                    "Şu halde", "O halde"]
NUM_VARIANTS = 9
NUM_WORKERS = 4
BERT_TOP_K = 50
MIN_REPLACEMENT_LENGTH = 3

@lru_cache(maxsize=10000)
def get_nouns_from_sentence(sentence):
    try:
        analysis = morphology.analyzeAndDisambiguate(sentence)
        nouns = set()
        for result in analysis.bestAnalysis():
            if result.getPos() and result.getPos().name() == "Noun":
                surface = result.surfaceForm()
                if len(surface) >= MIN_REPLACEMENT_LENGTH:
                    nouns.add(surface)
        return list(nouns)
    except Exception as e:
        print(f"Error in get_nouns_from_sentence: {str(e)}")
        return []

@lru_cache(maxsize=10000)
def get_stem(word):
    try:
        analysis = morphology.analyze(word)
        if analysis.isCorrect():
            return analysis.getAnalysisResults()[0].getStem()
    except Exception as e:
        print(f"Error in get_stem: {str(e)}")
        return None

def get_bert_candidates(target_word, top_k=10):
    try:
        mask_template = f"{target_word} veya [MASK] olabilir."
        results = fill_mask(mask_template, top_k=BERT_TOP_K)

        candidates = []
        seen = set()

        for res in results:
            candidate = res["token_str"].strip()

            if (len(candidate) < MIN_REPLACEMENT_LENGTH or
                candidate.lower() in seen or
                target_word.lower() in candidate.lower() or
                candidate.lower() in target_word.lower()):
                continue

            if not spell_checker.check(candidate):
                suggestions = spell_checker.suggestForWord(candidate)
                if suggestions:
                    candidate = suggestions[0]

            analysis = morphology.analyze(candidate)
            if (analysis.isCorrect() and
                analysis.getAnalysisResults()[0].getPos().name() == "Noun"):
                candidates.append(candidate)
                seen.add(candidate.lower())
                if len(candidates) >= top_k:
                    break

        return candidates
    except Exception as e:
        print(f"Error in get_bert_candidates: {str(e)}")
        return []

def generate_all_possible_replacements(nouns):
    replacements = {}
    for noun in nouns:
        candidates = get_bert_candidates(noun, top_k=10)
        if candidates:
            replacements[noun] = candidates
    return replacements

def generate_morph_forms(stem, features):
    try:
        items = morphology.getLexicon().getMatchingItems(stem)
        if not items.isEmpty():
            generation_results = morphology.getWordGenerator().generate(items.get(0), *features)
            return [result.surface for result in generation_results]
    except Exception as e:
        print(f"Error in generate_morph_forms: {str(e)}")
    return []

def get_morph_features(word):
    try:
        analysis = morphology.analyze(word)
        if analysis.isCorrect():
            return [ig.id for ig in analysis.getAnalysisResults()[0].getMorphemes()]
    except Exception as e:
        print(f"Error in get_morph_features: {str(e)}")
    return None

def generate_bert_variants(good_sentence, bad_sentence, nouns, replacements):
    variants = []
    used_combinations = set()

    for noun, candidates in replacements.items():
        for candidate in candidates:
            stem = get_stem(candidate)
            features = get_morph_features(noun)
            if not stem or not features:
                continue

            forms = generate_morph_forms(stem, features)
            if not forms:
                continue

            replacement = random.choice(forms)
            new_good = good_sentence.replace(noun, replacement, 1)
            new_bad = bad_sentence.replace(noun, replacement, 1)

            if (new_good, new_bad) not in variants:
                variants.append((new_good, new_bad))
                used_combinations.add((noun, candidate))

                if len(variants) >= NUM_VARIANTS:
                    return variants

    return variants

def generate_variants(good_sentence, bad_sentence, exclude_words):
    variants = []

    # Get all candidate nouns in good sentence
    nouns = [n for n in get_nouns_from_sentence(good_sentence)
             if n not in exclude_words]

    # 1: Try to get replacements BERT
    if nouns:
        replacements = generate_all_possible_replacements(nouns)
        if replacements:
            variants = generate_bert_variants(good_sentence, bad_sentence, nouns, replacements)

    # 2: If BERT didn't give enough candidates with success, try generating words
    if nouns and len(variants) < NUM_VARIANTS:
        for noun in nouns:
            if len(variants) >= NUM_VARIANTS:
                break

            features = get_morph_features(noun)
            if not features:
                continue

            stem = get_stem(noun)
            if not stem:
                continue

            items = morphology.getLexicon().getMatchingItems(stem)
            if items.isEmpty():
                continue

            generation_results = morphology.getWordGenerator().generate(items.get(0), *features)
            if generation_results:
                for result in generation_results:
                    if result.surface != noun:
                        new_good = good_sentence.replace(noun, result.surface, 1)
                        new_bad = bad_sentence.replace(noun, result.surface, 1)
                        if (new_good, new_bad) not in variants:
                            variants.append((new_good, new_bad))
                            if len(variants) >= NUM_VARIANTS:
                                break

    # 3: As last resort, use discourse markers at the beginning of the sentence
    if len(variants) < NUM_VARIANTS:
        needed = NUM_VARIANTS - len(variants)
        markers = random.sample(DISCOURSE_MARKERS, min(needed, len(DISCOURSE_MARKERS)))
        variants.extend([
            (f"{marker} {good_sentence[0].lower() + good_sentence[1:]}",
             f"{marker} {bad_sentence[0].lower() + bad_sentence[1:]}")
            for marker in markers
        ])

    return variants[:NUM_VARIANTS]

def process_file(file_path):
    try:
        print(f"\nStarting processing for {file_path}")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        df = pd.read_csv(file_path, sep=';')
        if df.empty:
            print(f"Empty dataframe for {file_path}")
            return None

        augmented_rows = []
        stats = {
            'total_sentences': len(df),
            'bert_variants': 0,
            'morph_variants': 0,
            'discourse_variants': 0
        }

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(file_path)}"):
            try:
                exclude_words = set()
                for col in ['good_cue', 'bad_cue', 'critical_region']:
                    if pd.notna(row[col]):
                        exclude_words.update(str(row[col]).split())

                variants = generate_variants(
                    str(row['good_sentence']),
                    str(row['bad_sentence']),
                    exclude_words
                )

                for good_var, bad_var in variants:
                    new_row = row.copy()
                    new_row['good_sentence'] = good_var
                    new_row['bad_sentence'] = bad_var
                    augmented_rows.append(new_row)

                    if any(marker in good_var for marker in DISCOURSE_MARKERS):
                        stats['discourse_variants'] += 1
                    else:
                        stats['bert_variants'] += 1

            except Exception as e:
                print(f"Error processing row {idx} in {file_path}: {str(e)}")
                traceback.print_exc()
                continue

        augmented_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

        output_path = f"augmented_{os.path.basename(file_path)}"
        augmented_df.to_csv(output_path, sep=';', index=False)

        print(f"\nCompleted {file_path}:")

        return output_path

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        traceback.print_exc()
        return None

def main():
    file_names = [
        'anaphor_agreement.csv',
        'argument_structure_ditransitive.csv',
        'argument_structure_transitive.csv',
        'binding.csv',
        'determiners.csv',
        'ellipsis.csv',
        'irregular_forms.csv',
        'island_effects.csv',
        'nominalization.csv',
        'npi_licensing.csv',
        'passives.csv',
        'quantifiers.csv',
        'relative_clauses.csv',
        'scrambling.csv',
        'subject_verb_agreement.csv',
        'suspended_affixation.csv',
    ]

    missing_files = [f for f in file_names if not os.path.exists(f)]
    if missing_files:
        print("Warning: The following files were not found:")
        for f in missing_files:
            print(f"- {f}")
        file_names = [f for f in file_names if os.path.exists(f)]

    successful_files = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_file, file_names), total=len(file_names)))

        for file, result in zip(file_names, results):
            if result:
                successful_files.append(result)
            else:
                print(f"Failed to process: {file}")

    print("\nAugmentation complete. Successfully processed files:")
    for result in successful_files:
        print(f"- {result}")

    if len(successful_files) < len(file_names):
        print("\nWarning: Some files could not be processed.")

if __name__ == "__main__":
    main()