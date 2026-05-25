from transformers import AutoTokenizer
import torch
from src.gpt_with_kv_caching.model import DecoderOnlyTransformer


def generate_next(input, model, kv_cache, use_kv_cache):
    with torch.no_grad():
        if use_kv_cache:
            logits, kv_cache = model(input_tokens=input, kv_cache=kv_cache)
        else:
            logits = model(input_tokens=input)

    next_logit = logits[:, -1:]
    next_token = torch.argmax(next_logit, dim=-1)
    return next_token, kv_cache


def generate(sequence, model_args, tokenizer_args, max_new_tokens):
    # Tokenizer
    tokenizer_path = tokenizer_args["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # Model
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=model_args["d_model"],
        d_ff=model_args["d_ff"],
        n_heads=model_args["n_heads"],
        n_layers=model_args["n_layers"],
        max_seq_len=model_args["max_seq_len"],
        dropout=model_args["dropout"],
        use_kv_cache=model_args["use_kv_cache"],
    )

    model.eval()
    tokenized_input = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
    input = tokenized_input["input_ids"]

    eos_token_id = tokenizer.eos_token_id
    use_kv_cache = model_args["use_kv_cache"]

    if use_kv_cache:
        # Prefill: full prompt seeds the cache and produces the first new token.
        next_token, kv_cache = generate_next(
            input=input, model=model, kv_cache=None, use_kv_cache=True
        )
        output = torch.cat((input, next_token), dim=-1)
        # Decode: feed only the new token each step.
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            next_token, kv_cache = generate_next(
                input=next_token, model=model, kv_cache=kv_cache, use_kv_cache=True
            )
            output = torch.cat((output, next_token), dim=-1)
    else:
        # No cache: re-run the full growing sequence each step.
        output = input
        for _ in range(max_new_tokens):
            next_token, _ = generate_next(
                input=output, model=model, kv_cache=None, use_kv_cache=False
            )
            output = torch.cat((output, next_token), dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    decoded_seq = tokenizer.decode(output[0])
    print(f"Starting sequence: {sequence}")
    print(f"Final Prediction: {decoded_seq}")
    return output


def sanity_check(sequence, model_args, tokenizer_args, max_new_tokens=5, seed=0):
    """Greedy decoding with and without KV cache must yield identical tokens.

    Re-seeding torch before each call gives both models the same random init;
    model.eval() disables dropout. Any divergence points to a bug in the cache
    path (positional offset, K/V concatenation, attention inputs, etc.).
    """
    args_cache = {**model_args, "use_kv_cache": True}
    args_no_cache = {**model_args, "use_kv_cache": False}

    torch.manual_seed(seed)
    tokens_no_cache = generate(sequence, args_no_cache, tokenizer_args, max_new_tokens)
    torch.manual_seed(seed)
    tokens_cache = generate(sequence, args_cache, tokenizer_args, max_new_tokens)

    if torch.equal(tokens_cache, tokens_no_cache):
        print(f"Sanity check PASSED — token IDs match: {tokens_cache[0].tolist()}")
    else:
        print("Sanity check FAILED — token IDs differ:")
        print(f"  no cache:   {tokens_no_cache[0].tolist()}")
        print(f"  with cache: {tokens_cache[0].tolist()}")
        raise AssertionError("KV-cache generation diverged from no-cache baseline")


if __name__ == "__main__":
    tokenizer_args = {"tokenizer_path": "t5-small"}

    model_args = {
        "d_model": 768,
        "d_ff": 3072,
        "n_heads": 12,
        "n_layers": 2,
        "max_seq_len": 512,
        "dropout": 0.1,
        "use_kv_cache": True,
    }

    sequence = "Hello my name is"

    generate(
        sequence=sequence,
        model_args=model_args,
        tokenizer_args=tokenizer_args,
        max_new_tokens=4,
    )

    # sanity_check(
    #     sequence=sequence,
    #     model_args=model_args,
    #     tokenizer_args=tokenizer_args,
    #     max_new_tokens=5,
    # )
