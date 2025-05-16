import math
import torch
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
import wandb

def load_model(tokenizer):
    """
    Creates a T5 model with the decoder start token set to pad_token_id.
    Resizes embeddings to match the tokenizer.
    """
    config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    return model

def preprocess_function(examples, tokenizer, max_length_sentence, task):
    """
    Depending on the task, pick different columns from the dataset
    for inputs/targets.
    """
    if task == "span_corruption":
        inputs = examples["input_sequence_span_corruption"]
        targets = examples["output_sequence_span_corruption"]
    elif task == "translation_pair_span_corruption":
        inputs = examples["input_sequence_translation_pair_span_corruption"]
        targets = examples["output_sequence_translation_pair_span_corruption"]
    elif task == "span_corruption_with_target_concat":
        inputs = examples["input_sequence_source_only_span_corruption_with_target_concat"]
        targets = examples["output_sequence_source_only_span_corruption_with_target_concat"]
    else:
        raise ValueError(f"Unsupported task: {task}")

    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=max_length_sentence,
        truncation=True
    )
    return model_inputs

def prepare_datasets_tsv(data_set, tokenizer, max_length_sentence, task):
    """
    Load train/validation/test from generic TSV. 
    """
    DATA_DIR = "path/to/data"
    data_files = {
        "train": f"{DATA_DIR}/{data_set}/{data_set}_{task}_train.tsv",
        "validation": f"{DATA_DIR}/{data_set}/{data_set}_{task}_validation.tsv",
        "test": f"{DATA_DIR}/{data_set}/{data_set}_{task}_test.tsv"
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

    for split in dataset.keys():
        dataset[split] = dataset[split].filter(
            lambda ex: all(col_val and col_val.strip() != "" for col_val in ex.values())
        )

    tokenized = dataset.map(
        lambda ex: preprocess_function(ex, tokenizer, max_length_sentence, task),
        batched=True
    )

    for split in tokenized.keys():
        keep_cols = {"input_ids", "attention_mask", "labels"}
        remove_cols = [c for c in tokenized[split].column_names if c not in keep_cols]
        tokenized[split] = tokenized[split].remove_columns(remove_cols)
        tokenized[split].set_format("torch")

    train_dataset = tokenized["train"]
    val_dataset = tokenized["validation"]
    test_dataset = tokenized["test"]
    return train_dataset, val_dataset, test_dataset

def prepare_training_objects(learning_rate, model, train_dataloader, eval_dataloader, test_dataloader):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )
    return optimizer, accelerator, model, optimizer, train_dataloader, eval_dataloader, test_dataloader

def prepare_dataloaders_pretrain(
    train_dataset,
    val_dataset,
    test_dataset,
    data_collator,
    train_batch_size,
    validation_batch_size
):
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_batch_size
    )
    eval_dataloader = DataLoader(
        val_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=validation_batch_size
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=validation_batch_size
    )
    return train_dataloader, eval_dataloader, test_dataloader

def inverse_square_root_schedule(optimizer, step, warmup_steps=1e4, init_lr=0.01):
    # Example of a custom LR schedule
    step = max(step, warmup_steps)
    lr = init_lr / math.sqrt(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train_step(model, dataloader, optimizer, accelerator, current_step):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        total_loss += loss.item()
        current_step += 1
    avg_train_loss = total_loss / len(dataloader)
    train_perplexity = torch.exp(torch.tensor(avg_train_loss))
    return {"loss": avg_train_loss, "perplexity": train_perplexity.item()}, current_step

def validation_step(model, dataloader):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Evaluation"):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
    avg_eval_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_eval_loss))
    return {"loss": avg_eval_loss, "perplexity": perplexity.item()}

def wandb_log_metrics(epoch, train_metrics, eval_metrics):
    wandb.log({
        "epoch/epoch": epoch,
        "loss/train_loss": train_metrics["loss"],
        "train/perplexity": train_metrics["perplexity"],
        "loss/eval_loss": eval_metrics["loss"],
        "eval/perplexity": eval_metrics["perplexity"]
    })

def train_model(
    model,
    optimizer,
    accelerator,
    train_dataloader,
    eval_dataloader,
    test_dataloader,
    num_train_epochs,
    run_name,
    patience
):
    num_training_steps = num_train_epochs * len(train_dataloader)
    print(f"Number of training steps: {num_training_steps}")

    lowest_loss = float("inf")
    early_stopping_counter = 0
    best_model_state = None
    current_step = 0

    for epoch in range(num_train_epochs):
        train_metrics, current_step = train_step(
            model, train_dataloader, optimizer, accelerator, current_step
        )
        eval_metrics = validation_step(model, eval_dataloader)
        wandb_log_metrics(epoch, train_metrics, eval_metrics)

        if eval_metrics["loss"] < lowest_loss:
            lowest_loss = eval_metrics["loss"]
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), f"{run_name}.pth")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered. No improvement in {patience} epochs.")
                break

    model.load_state_dict(best_model_state)
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(f"{run_name}.pth")
    wandb.log_artifact(artifact)
