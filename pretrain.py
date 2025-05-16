import argparse
import wandb

from shared_utils_generic import *
from pretrain_utils_generic import *

def main(
    num_train_epochs,
    max_length_sentence,
    train_batch_size,
    validation_batch_size,
    lr,
    data_set,
    comment,
    patience,
    task,
    local_tokenizer_path
):
    config, run_name = generate_config_and_run_name(
        num_train_epochs=num_train_epochs,
        data_set=data_set,
        comment=comment,
        task=task
    )

    wandb.init(project="Transformers-PALGA", config=config)
    tokenizer = load_tokenizer(local_tokenizer_path)
    assert tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be set."

    train_dataset, val_dataset, test_dataset = prepare_datasets_tsv(
        data_set, tokenizer, max_length_sentence, task
    )

    model = load_model(tokenizer)
    data_collator = prepare_datacollator(tokenizer, model)
    train_dl, eval_dl, test_dl = prepare_dataloaders_pretrain(
        train_dataset, val_dataset, test_dataset,
        data_collator, train_batch_size, validation_batch_size
    )

    optimizer, accelerator, model, optimizer, train_dl, eval_dl, test_dl = prepare_training_objects(
        lr, model, train_dl, eval_dl, test_dl
    )

    print(f"Train DataLoader Size: {len(train_dl)}")
    print(f"Eval DataLoader Size:  {len(eval_dl)}")
    print(f"Test DataLoader Size:  {len(test_dl)}")

    train_model(
        model,
        optimizer,
        accelerator,
        train_dl,
        eval_dl,
        test_dl,
        num_train_epochs,
        run_name,
        patience
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretraining parameters for a transformer model")

    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_length_sentence", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--validation_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--data_set", type=str, default="pretrain")
    parser.add_argument("--comment", type=str, default="test")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--task", type=str, default="span_corruption")
    parser.add_argument("--local_tokenizer_path", type=str, default="google/mT5-small")

    args = parser.parse_args()

    main(
        args.num_train_epochs,
        args.max_length_sentence,
        args.train_batch_size,
        args.validation_batch_size,
        args.learning_rate,
        args.data_set,
        args.comment,
        args.patience,
        args.task,
        args.local_tokenizer_path
    )
