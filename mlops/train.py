from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from src.data_utils.tokenizer_utils import Tokenizer
from src.encoder_decoder_transformer.model import EncoderDecoderTransformer
from torch import optim, nn


def load_dataset(data_args: dict, seed: int = 42) -> DatasetDict:
    """Load a line-delimited text file and split 70/20/10 into train/valid/test."""
    print("\nLoading and splitting dataset (Train/Valid/Test)")
    print(f"  Source: {data_args['dataset_path']}")
    with open(data_args["dataset_path"], "r") as f:
        text = f.read().splitlines()
    dataset = Dataset.from_list([{"text": sentence} for sentence in text])

    # 70% train, 30% held out
    train_heldout = dataset.train_test_split(test_size=0.3, seed=seed)

    # Split the 30% so that valid is 20% of total and test is 10% of total.
    # test_size = 10/30 = 1/3 of the held-out portion.
    valid_test = train_heldout["test"].train_test_split(test_size=1 / 3, seed=seed)

    split_dataset = DatasetDict(
        {
            "train": train_heldout["train"],
            "valid": valid_test["train"],
            "test": valid_test["test"],
        }
    )

    print(
        f"  Split sizes — Train: {len(split_dataset['train'])}, "
        f"Valid: {len(split_dataset['valid'])}, Test: {len(split_dataset['test'])}"
    )

    return split_dataset


def tokenize_dataset(
    dataset: DatasetDict, tokenizer: Tokenizer
) -> tuple[Dataset, Dataset, Dataset]:
    print("\nTokenizing splits (Train/Valid/Test)")
    train_set = tokenizer(dataset["train"])
    valid_set = tokenizer(dataset["valid"])
    test_set = tokenizer(dataset["test"])
    print(f"  Vocab size: {tokenizer.get_vocab_size():,}")
    print(f"  Columns after tokenization: {train_set.column_names}")
    return train_set, valid_set, test_set


def create_dataloader(
    train_set: Dataset,
    valid_set: Dataset,
    test_set: Dataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    print("\nBuilding DataLoaders (Train/Valid/Test)")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print(
        f"  Batch size: {batch_size} | "
        f"Train batches: {len(train_loader)}, "
        f"Valid batches: {len(valid_loader)}, "
        f"Test batches: {len(test_loader)}"
    )
    return train_loader, valid_loader, test_loader


def initialize_model(
    tokenizer: Tokenizer, model_args: dict
) -> EncoderDecoderTransformer:
    print("\nInitializing model")
    model = EncoderDecoderTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_args["d_model"],
        d_ff=model_args["d_ff"],
        n_heads=model_args["n_heads"],
        n_layers=model_args["n_layers"],
        max_seq_len=model_args["max_seq_len"],
        dropout=model_args["dropout"],
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Model: d_model={model_args['d_model']}, d_ff={model_args['d_ff']}, "
        f"heads={model_args['n_heads']}, layers={model_args['n_layers']}, "
        f"max_seq_len={model_args['max_seq_len']}, dropout={model_args['dropout']}"
    )
    print(f"  Params: {total:,} total ({trainable:,} trainable)")
    return model


def initialize_optim(model: nn.Module, optim_args: dict) -> optim.AdamW:
    print("\nInitializing optimizer")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optim_args["lr"],
        betas=optim_args["betas"],
        eps=optim_args["eps"],
        weight_decay=optim_args["weight_decay"],
    )
    print(
        f"  Optimizer: AdamW(lr={optim_args['lr']}, betas={optim_args['betas']}, "
        f"eps={optim_args['eps']}, weight_decay={optim_args['weight_decay']})"
    )
    return optimizer


def initialize_scheduler(
    optimizer: optim.Optimizer, scheduler_args: dict
) -> optim.lr_scheduler.LambdaLR:
    print("\nInitializing scheduler")
    # Linear warmup, then constant. Avoids the well-known early-training
    # instability transformers show without warmup. After `warmup_steps`,
    # the lr stays at the base `optim_args["lr"]`.
    warmup_steps = scheduler_args["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(f"  Scheduler: LambdaLR(linear warmup, warmup_steps={warmup_steps})")
    return scheduler


def initialize_criterion(
    tokenizer: Tokenizer, criterion_args: dict
) -> nn.CrossEntropyLoss:
    print("\nInitializing criterion")
    # pad/eos collision check — `ignore_index=pad_id` would also mask EOS
    # tokens in targets if the two share an id (GPT-2-style tokenizers).
    pad_id = tokenizer.tokenizer.pad_token_id
    eos_id = tokenizer.tokenizer.eos_token_id
    if pad_id == eos_id:
        raise ValueError(
            f"pad_token_id ({pad_id}) equals eos_token_id ({eos_id}); "
            "ignore_index=pad_id would also mask real EOS tokens in targets. "
            "Configure a distinct pad token for this tokenizer."
        )
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        label_smoothing=criterion_args["label_smoothing"],
    )
    print(
        f"  Criterion: CrossEntropyLoss(ignore_index={pad_id}, "
        f"label_smoothing={criterion_args['label_smoothing']})"
    )
    return criterion


def train_loop() -> None:
    return


def train(
    data_args: dict,
    tokenizer_args: dict,
    dataloader_args: dict,
    model_args: dict,
    optim_args: dict,
    scheduler_args: dict,
    criterion_args: dict,
) -> None:
    dataset = load_dataset(data_args)
    tokenizer = Tokenizer(
        tokenizer_path=tokenizer_args["tokenizer_path"],
        batch_size=tokenizer_args["batch_size"],
        padding=tokenizer_args["padding"],
        max_length=tokenizer_args["max_length"],
    )
    train_set, valid_set, test_set = tokenize_dataset(dataset, tokenizer)
    train_loader, valid_loader, test_loader = create_dataloader(
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        batch_size=dataloader_args["batch_size"],
    )
    # Peek at one batch so the shapes are visible before model init.
    batch = next(iter(train_loader))
    print("\nFirst training batch (shape sanity check):")
    for key, value in batch.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else f"len={len(value)}"
        print(f"  {key}: {shape}")

    model = initialize_model(tokenizer=tokenizer, model_args=model_args)
    optimizer = initialize_optim(model=model, optim_args=optim_args)
    scheduler = initialize_scheduler(optimizer=optimizer, scheduler_args=scheduler_args)
    criterion = initialize_criterion(tokenizer=tokenizer, criterion_args=criterion_args)


if __name__ == "__main__":
    data_args = {"dataset_path": "data/dataset.txt"}
    tokenizer_args = {
        "tokenizer_path": "t5-small",
        "batch_size": 100,
        "padding": "max_length",
        "max_length": 128,
    }
    dataloader_args = {"batch_size": 2}
    model_args = {
        "d_model": 512,
        "d_ff": 2048,
        "n_heads": 2,
        "n_layers": 6,
        "max_seq_len": 128,
        "dropout": 0.0,
    }  # EncoderDecoderTransformer Model
    optim_args = {
        "lr": 1e-4,
        "betas": (0.9, 0.98),
        "eps": 1e-9,
        "weight_decay": 0.0,
    }  # AdamW
    # betas=(0.9, 0.98) and eps=1e-9 are from "Attention Is All You Need".
    # weight_decay=0 keeps AdamW equivalent to paper-Adam; raise to 0.01 to
    # match common modern practice.
    scheduler_args = {"warmup_steps": 4000}
    criterion_args = {"label_smoothing": 0.1}

    train(
        data_args,
        tokenizer_args,
        dataloader_args,
        model_args,
        optim_args,
        scheduler_args,
        criterion_args,
    )

    print()
