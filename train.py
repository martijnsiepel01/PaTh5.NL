import argparse

# Note the generic import names now:
from finetune_utils_generic import *
from shared_utils_generic import *
from dag import create_palga_dag

def main(
    num_train_epochs,
    max_generate_length,
    train_batch_size,
    validation_batch_size,
    learning_rate,
    max_length_sentence,
    data_set,
    local_tokenizer_path,
    local_model_path,
    comment,
    patience,
    freeze_all_but_x_layers,
    lr_strategy,
    optimizer_type,
    dropout_rate,
    constrained_decoding
):
    config, run_name = generate_config_and_run_name(
        num_train_epochs=num_train_epochs,
        data_set=data_set,
        comment=comment
    )
    print(f"Run name: {run_name}")

    tokenizer = load_tokenizer(local_tokenizer_path)
    train_dataset, val_datasets = prepare_datasets_tsv(
        data_set, tokenizer, max_length_sentence
    )
    model = setup_model(
        tokenizer,
        freeze_all_but_x_layers,
        local_model_path,
        dropout_rate
    )
    data_collator = prepare_datacollator(tokenizer, model)
    train_dataloader, eval_dataloaders = prepare_dataloaders(
        train_dataset,
        val_datasets,
        data_collator,
        train_batch_size,
        validation_batch_size
    )

    num_training_steps = num_train_epochs * len(train_dataloader)

    optimizer, accelerator, model, train_dataloader, eval_dataloaders, scheduler = prepare_training_objects(
        learning_rate,
        model,
        train_dataloader,
        eval_dataloaders,
        lr_strategy,
        num_training_steps,
        optimizer_type
    )

    # Optionally build a dag for constrained decoding
    Palga_dag = None
    if constrained_decoding:
        # Provide your own file paths for these
        THESAURUS_PATH = "path/to/snomed_20230426.txt"
        DATA_LOCATION = "path/to/pretrain.tsv"
        EXCLUSIVE_TERMS_FILE = "path/to/mutually_exclusive_values.txt"
        Palga_dag = create_palga_dag(
            THESAURUS_PATH,
            local_tokenizer_path,
            DATA_LOCATION,
            EXCLUSIVE_TERMS_FILE
        )

    train_model(
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
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning parameters for a huggingface transformer model")

    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--max_generate_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--validation_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length_sentence", type=int, default=2048)
    parser.add_argument("--data_set", type=str, default="all")
    parser.add_argument("--local_tokenizer_path", type=str, default="google/mT5-small")
    parser.add_argument("--local_model_path", type=str, default="path/to/local_model")
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--freeze_all_but_x_layers", type=int, default=0)
    parser.add_argument("--lr_strategy", type=str, default="AdamW")
    parser.add_argument("--optimizer_type", type=str, default="AdamW")
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--constrained_decoding", action="store_true")

    args = parser.parse_args()

    main(
        args.num_train_epochs,
        args.max_generate_length,
        args.train_batch_size,
        args.validation_batch_size,
        args.learning_rate,
        args.max_length_sentence,
        args.data_set,
        args.local_tokenizer_path,
        args.local_model_path,
        args.comment,
        args.patience,
        args.freeze_all_but_x_layers,
        args.lr_strategy,
        args.optimizer_type,
        args.dropout_rate,
        args.constrained_decoding
    )
