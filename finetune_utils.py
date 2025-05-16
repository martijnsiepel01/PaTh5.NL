import torch
from tqdm import tqdm
from datasets import load_dataset
import wandb
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AdamW,
    Adafactor,
    AutoConfig,
    T5ForConditionalGeneration
)
import evaluate
from accelerate import Accelerator
import random

# Note the new import path:
from shared_utils_generic import *

# Change paths as needed
DATA_DIR = "path/to/data"
THESAURUS_PATH = "path/to/snomed_thesaurus.txt"
experiment_id = str(random.randint(1, 1_000_000))

def preprocess_function(examples, tokenizer, max_length_sentence):
    """
    Tokenize 'Conclusie_Conclusie' (input) and 'Codes' (target),
    also store sequence lengths for potential histogram-based splitting.
    """
    inputs = examples["Conclusie_Conclusie"]
    targets = examples["Codes"]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )
    lengths = [len(x.split()) for x in inputs]
    model_inputs["length"] = lengths
    return model_inputs

def prepare_datasets_tsv(data_set, tokenizer, max_length_sentence):
    """
    Load train/valid/test from TSV files, filter empty fields, tokenize,
    and split validation set by length.
    """
    data_files = {
        "train": f"{DATA_DIR}/{data_set}/train.tsv",
        "test": f"{DATA_DIR}/{data_set}/test.tsv",
        "validation": f"{DATA_DIR}/{data_set}/val.tsv"
    }

    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

    for split in dataset.keys():
        dataset[split] = dataset[split].filter(
            lambda ex: ex["Codes"] and ex["Codes"].strip()
        ).filter(
            lambda ex: ex["Conclusie_Conclusie"] and ex["Conclusie_Conclusie"].strip()
        )
        dataset[split] = dataset[split].map(
            lambda ex: preprocess_function(ex, tokenizer, max_length_sentence),
            batched=True
        )

        # Remove extra columns (keeping only input_ids, attention_mask, labels, length)
        keep_cols = {"input_ids", "attention_mask", "labels", "length"}
        remove_cols = [c for c in dataset[split].column_names if c not in keep_cols]
        dataset[split] = dataset[split].remove_columns(remove_cols)

    val_dataset = dataset["validation"].sort("length")
    total_len = len(val_dataset)
    split_sizes = [total_len // 5] * 4 + [total_len - (total_len // 5 * 4)]
    val_datasets = []
    start_idx = 0
    for size in split_sizes:
        end_idx = start_idx + size
        val_datasets.append(val_dataset.select(range(start_idx, end_idx)))
        start_idx = end_idx

    train_dataset = dataset["train"].remove_columns(["length"])
    val_datasets = [vd.remove_columns(["length"]) for vd in val_datasets]
    return train_dataset, val_datasets

def setup_model(
    tokenizer,
    freeze_all_but_x_layers,
    local_model_path="path/to/local/model",
    dropout_rate=0.1
):
    """
    Load or initialize a T5-based seq2seq model. Optionally set dropout rate.
    If you need partial freezing, implement it here.
    """
    config = AutoConfig.from_pretrained(local_model_path)
    if dropout_rate is not None:
        config.dropout_rate = dropout_rate

    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path, config=config)
    model.resize_token_embeddings(len(tokenizer))
    return model

def prepare_training_objects(
    learning_rate,
    model,
    train_dataloader,
    eval_dataloaders,
    lr_strategy,
    total_steps,
    optimizer_type="adamw"
):
    """
    Create optimizer, accelerator, and (optionally) schedule.
    """
    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == "adafactor":
        optimizer = Adafactor(
            model.parameters(),
            lr=1e-4,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
    else:
        raise ValueError("Unsupported optimizer type. Choose 'adamw' or 'adafactor'.")

    accelerator = Accelerator()
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    eval_dataloaders = [accelerator.prepare(dl) for dl in eval_dataloaders]

    # Currently not implementing a custom scheduler, return None
    scheduler = None
    return optimizer, accelerator, model, train_dataloader, eval_dataloaders, scheduler

# Load a thesaurus (e.g. SNOMED) to map codes -> words
thesaurus = pd.read_csv(THESAURUS_PATH, sep="|", encoding="latin-1")

def get_word_from_code(code):
    """
    Map a code to a human-readable word. If not found, return "Unknown".
    """
    if code.lower() in ["[c-sep]", "[c-sep]"]:
        return "[C-SEP]"
    matches = thesaurus[
        (thesaurus["DEPALCE"].str.lower() == code.lower()) &
        (thesaurus["DESTACE"] == "V")
    ]["DETEROM"].values
    return matches[0] if len(matches) > 0 else "Unknown"

def train_step(model, dataloader, optimizer, accelerator, scheduler, tokenizer):
    """
    Single epoch training loop. Example includes a reweighting step for [C-SEP].
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validation_step(model, eval_dataloaders, tokenizer, max_generate_length, constrained_decoding, Palga_dag):
    """
    Validate over 5 splits, returning metrics (BLEU, ROUGE, etc.).
    """
    import evaluate
    dataloader_names = ["shortest", "short", "average", "long", "longest"]
    if len(eval_dataloaders) != 5:
        raise ValueError("Must have exactly 5 validation dataloaders.")

    all_metrics = {}
    for dataloader, name in zip(eval_dataloaders, dataloader_names):
        metric_bleu = evaluate.load("sacrebleu", experiment_id=experiment_id)
        metric_rouge = evaluate.load("rouge", experiment_id=experiment_id)

        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc=f"Validating {name}"):
            if "length" in batch:
                del batch["length"]

            with torch.no_grad():
                if constrained_decoding and Palga_dag:
                    # Use dag-based constraints
                    prefix_allowed_tokens_fn = create_prefix_allowed_tokens_fn(Palga_dag, tokenizer)
                    outputs = model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=max_generate_length,
                        diversity_penalty=0.3,
                        num_beams=6,
                        num_beam_groups=2,
                        no_repeat_ngram_size=3,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
                    )
                else:
                    outputs = model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=max_generate_length,
                        diversity_penalty=0.3,
                        num_beams=6,
                        num_beam_groups=2,
                        no_repeat_ngram_size=3
                    )

                loss = model(**batch).loss
                total_loss += loss.item()

            filtered_preds = [
                [tid for tid in seq if tid != tokenizer.pad_token_id]
                for seq in outputs
            ]
            filtered_labels = [
                [tid for tid in seq if tid != -100]
                for seq in batch["labels"]
            ]
            decoded_preds = [
                tokenizer.decode(pred, skip_special_tokens=True)
                for pred in filtered_preds
            ]
            decoded_labels = [
                tokenizer.decode(label, skip_special_tokens=True)
                for label in filtered_labels
            ]

            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)

        bleu = metric_bleu.compute(
            predictions=all_preds,
            references=[[lbl] for lbl in all_labels]
        )
        rouge_scores = metric_rouge.compute(
            predictions=all_preds,
            references=all_labels
        )

        ROUGE_1 = rouge_scores["rouge1"]
        ROUGE_2 = rouge_scores["rouge2"]
        ROUGE_L = rouge_scores["rougeL"]
        ROUGE_Lsum = rouge_scores["rougeLsum"]
        bleu_score = bleu["score"] / 100
        avg_rouge = (ROUGE_1 + ROUGE_2 + ROUGE_L + ROUGE_Lsum) / 4
        epsilon = 1e-7
        bleu_rouge_f1 = (2 * bleu_score * avg_rouge) / (bleu_score + avg_rouge + epsilon)

        all_metrics[f"loss_{name}"] = total_loss / len(dataloader)
        all_metrics[f"bleu_{name}"] = bleu_score
        all_metrics[f"average_rouge_{name}"] = avg_rouge
        all_metrics[f"bleu_rouge_f1_{name}"] = bleu_rouge_f1

    return all_metrics, [], [], []

def create_prefix_allowed_tokens_fn(Palga_dag, tokenizer):
    """
    Example function that returns which tokens are allowed next
    given a partial generation, using a dag.
    """
    c_sep_token_id = tokenizer.encode("[C-SEP]", add_special_tokens=False)

    def prefix_allowed_tokens_fn(batch_id, sent):
        sent_list = sent.tolist()[1:]
        last_index = -1

        if len(c_sep_token_id) > 1:
            for i in range(len(sent_list) - len(c_sep_token_id) + 1):
                if sent_list[i:i+len(c_sep_token_id)] == c_sep_token_id:
                    last_index = i + len(c_sep_token_id) - 1
        else:
            single_id = c_sep_token_id[0] if c_sep_token_id else None
            if single_id in sent_list:
                rev_idx = sent_list[::-1].index(single_id)
                last_index = len(sent_list) - 1 - rev_idx

        if last_index != -1:
            sent_list = sent_list[last_index+1:]

        allowed = list(Palga_dag.get(sent_list))
        return allowed if allowed else [tokenizer.eos_token_id]

    return prefix_allowed_tokens_fn

def wandb_log_metrics(epoch, train_loss, eval_metrics, suffix="", run=None):
    """
    Aggregate the 5 splits (shortest->longest) and log to W&B.
    """
    avg_eval_loss = (
        eval_metrics["loss_shortest"] +
        eval_metrics["loss_short"] +
        eval_metrics["loss_average"] +
        eval_metrics["loss_long"] +
        eval_metrics["loss_longest"]
    ) / 5

    avg_eval_bleu = (
        eval_metrics["bleu_shortest"] +
        eval_metrics["bleu_short"] +
        eval_metrics["bleu_average"] +
        eval_metrics["bleu_long"] +
        eval_metrics["bleu_longest"]
    ) / 5

    avg_eval_rouge = (
        eval_metrics["average_rouge_shortest"] +
        eval_metrics["average_rouge_short"] +
        eval_metrics["average_rouge_average"] +
        eval_metrics["average_rouge_long"] +
        eval_metrics["average_rouge_longest"]
    ) / 5

    avg_eval_f1 = (
        eval_metrics["bleu_rouge_f1_shortest"] +
        eval_metrics["bleu_rouge_f1_short"] +
        eval_metrics["bleu_rouge_f1_average"] +
        eval_metrics["bleu_rouge_f1_long"] +
        eval_metrics["bleu_rouge_f1_longest"]
    ) / 5

    metrics = {
        f"epoch/epoch{suffix}": epoch,
        f"training/train_loss{suffix}": train_loss,
        f"shortest/eval_loss_shortest{suffix}": eval_metrics["loss_shortest"],
        f"short/eval_loss_short{suffix}": eval_metrics["loss_short"],
        f"average/eval_loss_average{suffix}": eval_metrics["loss_average"],
        f"long/eval_loss_long{suffix}": eval_metrics["loss_long"],
        f"longest/eval_loss_longest{suffix}": eval_metrics["loss_longest"],
        f"shortest/eval_BLEU_shortest{suffix}": eval_metrics["bleu_shortest"],
        f"short/eval_BLEU_short{suffix}": eval_metrics["bleu_short"],
        f"average/eval_BLEU_average{suffix}": eval_metrics["bleu_average"],
        f"long/eval_BLEU_long{suffix}": eval_metrics["bleu_long"],
        f"longest/eval_BLEU_longest{suffix}": eval_metrics["bleu_longest"],
        f"shortest/eval_average_ROUGE_shortest{suffix}": eval_metrics["average_rouge_shortest"],
        f"short/eval_average_ROUGE_short{suffix}": eval_metrics["average_rouge_short"],
        f"average/eval_average_ROUGE_average{suffix}": eval_metrics["average_rouge_average"],
        f"long/eval_average_ROUGE_long{suffix}": eval_metrics["average_rouge_long"],
        f"longest/eval_average_ROUGE_longest{suffix}": eval_metrics["average_rouge_longest"],
        f"shortest/eval_F1-Bleu-Rouge_shortest{suffix}": eval_metrics["bleu_rouge_f1_shortest"],
        f"short/eval_F1-Bleu-Rouge_short{suffix}": eval_metrics["bleu_rouge_f1_short"],
        f"average/eval_F1-Bleu-Rouge_average{suffix}": eval_metrics["bleu_rouge_f1_average"],
        f"long/eval_F1-Bleu-Rouge_long{suffix}": eval_metrics["bleu_rouge_f1_long"],
        f"longest/eval_F1-Bleu-Rouge_longest{suffix}": eval_metrics["bleu_rouge_f1_longest"],
        f"evaluation/avg_loss{suffix}": avg_eval_loss,
        f"evaluation/avg_bleu{suffix}": avg_eval_bleu,
        f"evaluation/avg_rouge{suffix}": avg_eval_rouge,
        f"evaluation/avg_f1_bleu_rouge{suffix}": avg_eval_f1
    }

    if run:
        run.log(metrics)
    else:
        wandb.log(metrics)

def train_model(
    model,
    optimizer,
    accelerator,
    max_generate_length,
    train_dataloader,
    eval_dataloaders,
    num_train_epochs,
    tokenizer,
    run_name,
    patience,
    scheduler,
    Palga_dag,
    config,
    constrained_decoding
):
    """
    Main training loop with W&B logging and early stopping.
    """
    run = wandb.init(project="Transformers-PALGA", config=config, name=run_name, reinit=True)

    lowest_loss = float("inf")
    early_stopping_counter = 0
    best_model_state = None

    for epoch in range(num_train_epochs):
        avg_train_loss = train_step(
            model, train_dataloader, optimizer, accelerator, scheduler, tokenizer
        )
        eval_metrics, _, _, _ = validation_step(
            model, eval_dataloaders, tokenizer, max_generate_length, constrained_decoding, Palga_dag
        )
        wandb_log_metrics(epoch, avg_train_loss, eval_metrics, run=run)

        total_eval_loss = sum(
            eval_metrics[f"loss_{name}"] for name in ["shortest","short","average","long","longest"]
        )
        average_eval_loss = total_eval_loss / 5

        if average_eval_loss < lowest_loss:
            lowest_loss = average_eval_loss
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), f"{run_name}.pth")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break

    # Final validation with the best model
    model.load_state_dict(best_model_state)
    final_eval_metrics, _, _, _ = validation_step(
        model, eval_dataloaders, tokenizer, max_generate_length, constrained_decoding, Palga_dag
    )
    print("Final validation metrics:", final_eval_metrics)

    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(f"{run_name}.pth")
    run.log_artifact(artifact)
    run.finish()
