"""Positional Encoder.

Modules:
    - PositionalEncoder class
"""
import torch.nn as nn
import torch

class PositionalEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.0):
        """Initialize the positional encoder.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model.
            max_seq_len (int): Maximum sequence length.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Token Embedding Layer
        self.token_embedder = nn.Embedding(vocab_size, d_model)

        # Learned Positional Embedding Layer
        self.pos_embedder = nn.Embedding(max_seq_len, d_model)

        # scale
        self.scale = d_model ** 0.5

    def forward(self, tokens: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """Forward pass for the positional encoder.

        Args:
            tokens (torch.Tensor): Input tensor.
                Shape: (batch_size, seq_len)
            position_offset (int, optional): Starting absolute position for the
                first token in ``tokens``. Non-zero during KV-cache decode, when
                the input is a single new token sitting at position S of the
                full sequence rather than position 0.

        Returns:
            torch.Tensor: Output tensor.
                Shape: (batch_size, seq_len, d_model)
        """
        # Get the shape
        _, seq_len = tokens.size()

        # Get the token embeddings and scale
        token_embeddings = self.token_embedder(tokens) * self.scale

        # Get the positional embeddings - 1, seq_len
        positional_ids = torch.arange(
            position_offset, position_offset + seq_len, device=tokens.device
        ).unsqueeze(0)

        # Get the positional embeddings - 1, seq_len, d_model
        pos_embeddings = self.pos_embedder(positional_ids)

        # Add the token and positional embeddings
        embeddings = token_embeddings + pos_embeddings

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings # (batch_size, seq_len, d_model)

if __name__ == "__main__":
    vocab_size, d_model, max_seq_len = 10, 4, 12
    # create a positional encoder
    positional_encoder = PositionalEncoder(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)

    # create a test tensor
    tokens = torch.randint(0, vocab_size, (1, max_seq_len))

    # forward pass
    embeddings = positional_encoder(tokens)
    print(embeddings.shape)
    print(embeddings)
    print()