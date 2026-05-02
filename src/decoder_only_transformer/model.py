import torch.nn as nn
from src.utils.positional_encoder import PositionalEncoder
from src.decoder_only_transformer.decoder import DecoderStack
import torch


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        """Initialize the decoder only transformer.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed forward layer.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of decoder layers.
            max_seq_len (int): Maximum sequence length.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        # Positional Embedding Layers
        self.decoder_embed_layer = PositionalEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Decoder Stack
        self.decoder = DecoderStack(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Final output projection layer
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_tokens, attention_mask=None):
        """Forward pass for the decoder only transformer.

        Args:
            input_tokens (torch.Tensor): Input tensor to the decoder layer.
            attention_mask (torch.Tensor): Padding mask for the decoder input.
        """
        # Convert attention masks to padding masks and reshape to (B, 1, 1, K)
        # so they broadcast over (B, n_heads, Q, K) attention scores.
        padding_mask = (
            (~attention_mask.bool()).unsqueeze(1).unsqueeze(2)
            if attention_mask is not None
            else None
        )

        # Embed the input tokens
        input_embed = self.decoder_embed_layer(input_tokens)

        # Decode the input tokens
        decoder_output = self.decoder(
            decoder_input=input_embed, decoder_padding_mask=padding_mask
        )

        # Project the decoder output to the output vocabulary
        output = self.output_proj(decoder_output)
        return output


if __name__ == "__main__":
    vocab_size, d_model, d_ff, n_heads, n_layers, max_seq_len, dropout = (
        10,
        5,
        10,
        1,
        2,
        12,
        0.1,
    )
    # create a encoder decoder transformer
    decoder_only_transformer = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    # create a test tensor
    input_tokens = torch.randint(0, vocab_size, (1, max_seq_len))
    attention_mask = None

    # forward pass
    output = decoder_only_transformer(
        input_tokens=input_tokens, attention_mask=attention_mask
    )
    print(output.shape)
    print(output)
