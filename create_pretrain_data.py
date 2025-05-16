import pandas as pd
import numpy as np
import random

# Example: replace these with your own TSV file paths
FILE_PATHS = [
    "path/to/dataset1.tsv",
    "path/to/dataset2.tsv",
    "path/to/dataset3.tsv",
]

# Where to save the combined output
OUTPUT_FILE = "path/to/output/pretrain_combined_validation_data.tsv"

def span_corruption(sentence, mean_span_length=3, mask_rate=0.15, seed=None, start_extra_id=0):
    """
    Corrupts a sentence by masking spans of words with <extra_id_x> tokens.
    Returns:
      - corrupted_sentence: 'sentence' with masked spans replaced by <extra_id_x>
      - target_sentence: the sequence of masked spans with preceding <extra_id_x>
      - extra_id_counter: how many <extra_id_x> tokens were used
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    words = sentence.split()
    num_words = len(words)
    spans = []
    target_spans = []
    i = 0
    extra_id_counter = start_extra_id

    while i < num_words:
        if random.random() < mask_rate:
            # Draw a Poisson-distributed span length
            span_length = max(1, np.random.poisson(mean_span_length))
            span_end = min(i + span_length, num_words)

            spans.append(f"<extra_id_{extra_id_counter}>")
            target_spans.append(
                f"<extra_id_{extra_id_counter}> {' '.join(words[i:span_end])}"
            )

            i = span_end
            extra_id_counter += 1
        else:
            spans.append(words[i])
            i += 1

    # Append an additional <extra_id_x> to mark the end of the target sequence
    target_spans.append(f"<extra_id_{extra_id_counter}>")
    extra_id_counter += 1

    corrupted_sentence = " ".join(spans)
    target_sentence = " ".join(target_spans)
    return corrupted_sentence, target_sentence, extra_id_counter

def translation_pair_span_corruption(
    source_sentence, 
    target_sentence, 
    mean_span_length=3, 
    mask_rate=0.15, 
    seed=None
):
    """
    Corrupt both source and target sentences with span masking and
    concatenate them (source + target) for the model's input, and
    do the same for the masked spans for the model's output.
    """
    corrupted_source, target_source, next_extra_id = span_corruption(
        source_sentence, mean_span_length, mask_rate, seed, start_extra_id=0
    )
    corrupted_target, target_target, _ = span_corruption(
        target_sentence, mean_span_length, mask_rate, seed, start_extra_id=next_extra_id
    )
    model_input = corrupted_source + " " + corrupted_target
    model_output = target_source + " " + target_target
    return model_input, model_output

def source_only_span_corruption_with_target_concat(
    source_sentence, 
    target_sentence, 
    mean_span_length=3, 
    mask_rate=0.15, 
    seed=None
):
    """
    Corrupt only the source sentence, then concatenate the unmodified
    target sentence. Output is just the masked spans from the source.
    """
    corrupted_source, target_source, _ = span_corruption(
        source_sentence, mean_span_length, mask_rate, seed, start_extra_id=0
    )
    model_input = corrupted_source + " " + target_sentence
    model_output = target_source
    return model_input, model_output

def main():
    """
    Load multiple TSVs, generate corrupted versions of (source, target) pairs,
    and save the combined data to a single TSV with extra columns for each
    corruption variant.
    """
    dfs = []
    for file_path in FILE_PATHS:
        df = pd.read_csv(file_path, sep="\t")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    headers = [
        "source_sentence",
        "target_sentence",
        "input_sequence_span_corruption",
        "output_sequence_span_corruption",
        "input_sequence_translation_pair_span_corruption",
        "output_sequence_translation_pair_span_corruption",
        "input_sequence_source_only_span_corruption_with_target_concat",
        "output_sequence_source_only_span_corruption_with_target_concat"
    ]
    data = []

    for _, row in combined_df.iterrows():
        source_sentence = str(row.get("Conclusie", "")).strip()
        target_sentence = str(row.get("Codes", "")).strip()

        # Skip if source or target is empty or 'nan'
        if not source_sentence or source_sentence.lower() == "nan":
            continue
        if not target_sentence or target_sentence.lower() == "nan":
            continue

        in_span, out_span, _ = span_corruption(source_sentence, seed=1)
        in_trans, out_trans = translation_pair_span_corruption(
            source_sentence, target_sentence, seed=1
        )
        in_srconly, out_srconly = source_only_span_corruption_with_target_concat(
            source_sentence, target_sentence, seed=1
        )

        data.append([
            source_sentence,
            target_sentence,
            in_span,
            out_span,
            in_trans,
            out_trans,
            in_srconly,
            out_srconly
        ])

    data_df = pd.DataFrame(data, columns=headers)
    data_df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Saved combined pretraining data to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
