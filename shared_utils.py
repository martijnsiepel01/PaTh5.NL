from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, AutoTokenizer

def load_tokenizer(local_tokenizer_path):
    """
    Loads a tokenizer from a local path or pretrained name, ensuring '[C-SEP]' is recognized.
    """
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
    if "[C-SEP]" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["[C-SEP]"])
    return tokenizer

def prepare_datacollator(tokenizer, model):
    return DataCollatorForSeq2Seq(tokenizer, model=model)

def prepare_dataloaders(train_dataset, val_datasets, data_collator, train_batch_size, validation_batch_size):
    """
    Create a train DataLoader and a list of validation DataLoaders
    (for histogram-based or multiple-split evaluation).
    """
    from torch.utils.data import DataLoader

    train_dl = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_batch_size
    )
    eval_dls = [
        DataLoader(
            ds,
            collate_fn=data_collator,
            batch_size=validation_batch_size
        ) for ds in val_datasets
    ]
    return train_dl, eval_dls

def generate_config_and_run_name(**kwargs):
    """
    Build a config dict from kwargs and a run_name by concatenating key-value pairs.
    """
    config = dict(kwargs)
    run_name_parts = [f"{k}{v}" for k, v in kwargs.items()]
    run_name = "_".join(run_name_parts)
    return config, run_name
