"""Tokenizer wrapper around HuggingFace AutoTokenizer."""

from collections.abc import Mapping
from typing import Any, Optional, Union

import torch
from datasets import Dataset
from transformers import AutoTokenizer


class Tokenizer:
    """Thin wrapper around a HuggingFace tokenizer with a single
    `tokenize` entry point that accepts a string, a list of strings,
    or an HF Dataset.

    Mode is declared at construction. If `source_language` and
    `target_language` are set, the tokenizer operates in translation
    mode and expects Datasets to have a "translation" column. Otherwise
    it operates in text mode and expects a "text" column.
    """

    def __init__(
        self,
        tokenizer_path: str,
        batch_size: Optional[int] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        padding: str = "longest",
        truncation: bool = True,
        max_length: Optional[int] = None,
        num_proc: Optional[int] = None,
    ):
        """Initialize the tokenizer.

        Args:
            tokenizer_path (str): HF tokenizer name or local path.
            batch_size (int): Batch size for `Dataset.map()`. Default: None.
            source_language (str): Key for translation datasets (e.g. "en").
                Setting this (with `target_language`) puts the tokenizer in
                translation mode.
            target_language (str): Key for translation datasets (e.g. "de").
            padding (str): HF padding strategy.
                Default "longest" (dynamic, per-batch).
                Use "max_length" only when `max_length` is set, otherwise HF
                silently pads to the model's max length.
            truncation (bool): Whether to truncate beyond `max_length`.
            max_length (int): Maximum sequence length.
                Required when `padding="max_length"`.
            num_proc (int): Processes to use for `Dataset.map()`.
        """
        if padding == "max_length" and max_length is None:
            raise ValueError(
                "padding='max_length' requires an explicit max_length; "
                "otherwise HF falls back to the model's max length and "
                "silently inflates memory."
            )

        if (source_language is None) != (target_language is None):
            raise ValueError(
                "source_language and target_language must be set together."
            )

        self.batch_size = batch_size
        self.source_language = source_language
        self.target_language = target_language
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.num_proc = num_proc

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # GPT-2-style tokenizers have no pad token; reuse EOS.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def is_translation(self) -> bool:
        return self.source_language is not None

    def __call__(
        self, data: Union[str, list[str], Dataset]
    ) -> Union[dict[str, torch.Tensor], Dataset]:
        return self.tokenize(data)

    def tokenize(
        self, data: Union[str, list[str], Dataset]
    ) -> Union[dict[str, torch.Tensor], Dataset]:
        """Tokenize a string, a list of strings, or an HF Dataset.

        Returns a `dict[str, Tensor]` for strings/lists, or a Dataset
        with `.with_format("torch")` so rows yield tensors on access.
        """
        if isinstance(data, str) or (
            isinstance(data, list) and all(isinstance(x, str) for x in data)
        ):
            return self._tokenize_strings(data)

        if isinstance(data, Dataset):
            return self._tokenize_dataset(data)

        raise TypeError(
            f"Unsupported input type for tokenize(): {type(data).__name__}. "
            "Expected str, list[str], or datasets.Dataset."
        )

    def _tokenize_strings(self, text: Union[str, list[str]]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
        )

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        if self.is_translation:
            if "translation" not in dataset.column_names:
                raise ValueError(
                    "Translation mode requires a 'translation' column; "
                    f"got columns: {dataset.column_names}."
                )
            batch_fn = self._tokenize_translation_batch
        else:
            if "text" not in dataset.column_names:
                raise ValueError(
                    "Text mode requires a 'text' column; "
                    f"got columns: {dataset.column_names}."
                )
            batch_fn = self._tokenize_text_batch

        mapped = dataset.map(
            batch_fn,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
        )
        # Return tensors on access. `output_all_columns=True` keeps
        # non-numeric columns (e.g. the original "text" or "translation")
        # accessible as-is.
        return mapped.with_format("torch", output_all_columns=True)

    def _tokenize_translation_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        translations = batch["translation"]
        src = [example[self.source_language] for example in translations]
        tgt = [example[self.target_language] for example in translations]

        return self.tokenizer(
            src,
            text_target=tgt,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
        )

    def _tokenize_text_batch(self, batch: dict[str, list[str]]) -> dict[str, Any]:
        return self.tokenizer(
            batch["text"],
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
        )

    def detokenize(
        self,
        data: Union[
            Mapping[str, Any],
            Dataset,
            torch.Tensor,
            list[int],
            list[list[int]],
        ],
    ) -> Union[list[str], dict[str, list[str]]]:
        """Decode token ids back to strings.

        Accepts:
          * `Mapping` (dict / `BatchEncoding`) with an `"input_ids"` key.
          * `Dataset` with an `"input_ids"` column.
          * `torch.Tensor` of shape [T] or [B, T], or `list[int]` /
            `list[list[int]]` of raw token ids.

        Returns `list[str]` for the input side. If the input also carries
        `"labels"` (translation output from a Dataset), returns
        `{"source": [...], "target": [...]}` with both sides decoded.
        """
        labels = None

        # `Mapping` covers plain dicts and HF's `BatchEncoding`, which is
        # a MutableMapping but not a dict subclass.
        if isinstance(data, Mapping):
            input_ids = data["input_ids"]
            if "labels" in data:
                labels = data["labels"]
        elif isinstance(data, Dataset):
            # `dataset[col]` returns a `Column` (lazy list-like) that
            # `batch_decode` doesn't handle — materialize to a plain list.
            input_ids = list(data["input_ids"])
            if "labels" in data.column_names:
                labels = list(data["labels"])
        else:
            input_ids = data

        decoded_source = self._decode_ids(input_ids)
        if labels is None:
            return decoded_source
        return {"source": decoded_source, "target": self._decode_ids(labels)}

    def _decode_ids(
        self,
        input_ids: Union[torch.Tensor, list[int], list[list[int]]],
    ) -> list[str]:
        # Single sequence → wrap in a list so the return type is uniform.
        if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
            return [self.tokenizer.decode(input_ids, skip_special_tokens=True)]
        if (
            isinstance(input_ids, list)
            and len(input_ids) > 0
            and isinstance(input_ids[0], int)
        ):
            return [self.tokenizer.decode(input_ids, skip_special_tokens=True)]
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size


if __name__ == "__main__":
    tokenizer_path = "t5-small"

    sentence = "Hello, world! This is a test sentence."
    sentence_list = [
        "Hello, world! This is a test sentence.",
        "This is another test sentence.",
    ]

    tokenizer = Tokenizer(
        tokenizer_path=tokenizer_path,
        padding="max_length",
        max_length=20,
    )

    # Single string
    tokenized_input = tokenizer(sentence)
    detokenized = tokenizer.detokenize(tokenized_input)

    print(f"Tokenizer: {tokenizer_path}")
    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size():,}")
    print()
    print(f"Sentence: {sentence}")
    print(f"Tokenized input: {tokenized_input}")
    print(f"Detokenized input: {detokenized}")
    print()

    # List of strings
    tokenized_list = tokenizer(sentence_list)
    detokenized_list = tokenizer.detokenize(tokenized_list)
    print(f"Sentence list: {sentence_list}")
    print(f"Tokenized list: {tokenized_list}")
    print(f"Detokenized list: {detokenized_list}")
    print()

    # Text dataset
    dataset = Dataset.from_list([{"text": s} for s in sentence_list])
    tokenized_dataset = tokenizer(dataset)
    detokenized_dataset = tokenizer.detokenize(tokenized_dataset)

    print(f"Dataset: {dataset}")
    print(f"Tokenized dataset: {tokenized_dataset}")
    print(f"Decoded dataset: {detokenized_dataset}")
    print()

    # Translation dataset
    translation_tokenizer = Tokenizer(
        tokenizer_path=tokenizer_path,
        source_language="en",
        target_language="de",
        padding="max_length",
        max_length=20,
    )
    translation_dataset = Dataset.from_list(
        [
            {"translation": {"en": "Hello, world!", "de": "Hallo, Welt!"}},
            {"translation": {"en": "Good morning.", "de": "Guten Morgen."}},
        ]
    )
    tokenized_translation = translation_tokenizer(translation_dataset)
    decoded_translation_0 = translation_tokenizer.detokenize(tokenized_translation[:1])
    decoded_translation = translation_tokenizer.detokenize(tokenized_translation[:])
    print(f"Translation dataset: {translation_dataset}")
    print(f"Tokenized translation: {tokenized_translation}")
    print(f"Decoded row 0: {decoded_translation_0}")
    print(f"Decoded row 1: {decoded_translation}")

    print()
