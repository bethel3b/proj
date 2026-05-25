from src.gpt.model import DecoderOnlyTransformer
from transformers import AutoTokenizer
import torch


def generate_next(input, model):
    with torch.no_grad():
        logits = model(input_tokens=input)

    next_logit = logits[:, -1:]
    next_token = torch.argmax(next_logit, dim=-1)
    return next_token


def generate(sequence, model_args, tokenizer_args, max_new_tokens):
    tokenizer_path = tokenizer_args["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=model_args["d_model"],
        d_ff=model_args["d_ff"],
        n_heads=model_args["n_heads"],
        n_layers=model_args["n_layers"],
        max_seq_len=model_args["max_seq_len"],
        dropout=model_args["dropout"],
    )

    model.eval()
    tokenized_input = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
    input = tokenized_input["input_ids"]

    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        next_token = generate_next(input=input, model=model)
        input = torch.cat((input, next_token), dim=-1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    decoded_seq = tokenizer.decode(input[0])
    print(f"Starting sequence: {sequence}")
    print(f"Final Prediction: {decoded_seq}")
    return decoded_seq


if __name__ == "__main__":
    tokenizer_args = {"tokenizer_path": "t5-small"}

    model_args = {
        "d_model": 768,
        "d_ff": 3072,
        "n_heads": 12,
        "n_layers": 12,
        "max_seq_len": 512,
        "dropout": 0.1,
    }

    sequence = "Hello my name is"

    generate(
        sequence=sequence,
        model_args=model_args,
        tokenizer_args=tokenizer_args,
        max_new_tokens=4,
    )
