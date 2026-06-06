from src.utils.utils import print_header
from src.gpt_with_kv_caching.model import DecoderOnlyTransformer


def build_gpt_model(vocab_size: int, model_args: dict) -> DecoderOnlyTransformer:
    """Build the decoder-only transformer (GPT) and report its parameter count."""
    print_header(text="Initializing model")
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
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
        f"Model: d_model={model_args['d_model']}, d_ff={model_args['d_ff']}, "
        f"heads={model_args['n_heads']}, layers={model_args['n_layers']}, "
        f"max_seq_len={model_args['max_seq_len']}, dropout={model_args['dropout']}"
    )
    print(f"Params: {total:,} total ({trainable:,} trainable)")
    return model
