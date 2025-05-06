import torch
from torch import nn
from typing import Optional


# NOTE: there still no position embedding in the native transformer encoder
class NativeTransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        num_layers,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                activation="gelu",
                batch_first=True,
                dropout=dropout,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        v_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        @param x : (batch_size, seq_len, hidden_size)
        @param v_lengths : (batch_size, seq_len)
        """

        if v_lengths is not None:
            # Create a mask for the padding tokens
            video_padding_mask = self.create_key_padding_mask(
                v_lengths, max_len=x.size(1)
            )
        else:
            video_padding_mask = None

        for layer in self.encoder.layers:
            x = layer(x, src_key_padding_mask=video_padding_mask)

        return x, v_lengths, video_padding_mask

    @staticmethod
    def create_key_padding_mask(sequence_lengths, max_len=None):
        """
        Create a key padding mask from sequence lengths.

        Args:
            sequence_lengths: 1D tensor of sequence lengths (batch_size,)
            max_len: Maximum sequence length. If None, uses max(sequence_lengths)

        Returns:
            mask: BoolTensor of shape (batch_size, max_len) where False means padding
        """
        if max_len is None:
            max_len = sequence_lengths.max()
        batch_size = sequence_lengths.size(0)
        mask = torch.arange(max_len, device=sequence_lengths.device).expand(
            batch_size, max_len
        ) >= sequence_lengths.unsqueeze(1)
        return mask


if __name__ == "__main__":
    # Example usage
    model = NativeTransformerEncoder(
        hidden_size=1024,
        num_attention_heads=8,
        intermediate_size=2048,
        num_layers=6,
    ).cuda()
    x = torch.randn(2, 10, 1024).cuda()  # (batch_size, seq_len, hidden_size)
    v_lengths = torch.tensor([5, 8]).cuda()
    output = model(x, v_lengths)
    print(output.shape)  # Should be (2, 30, 1024)
