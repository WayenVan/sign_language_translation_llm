import torch
from torch import nn
from typing import Optional
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder


class SignBertEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        num_layers,
        dropout=0.1,
        max_position_embeddings=512,
    ) -> None:
        super().__init__()
        self.llm_config = self._generate_config(
            hidden_size, num_attention_heads, intermediate_size, num_layers
        )
        self.encoder = BertEncoder(self.llm_config)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)

    @staticmethod
    def _generate_config(
        hidden_size, num_attention_heads, intermediate_size, num_layers
    ):
        config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_layers,
            intermediate_size=intermediate_size,
            hidden_act="gelu",
            position_embedding_type="absolute",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            pad_token_id=None,
            max_position_embeddings=None,
            type_vocab_size=None,
            initializer_range=0.02,
            use_cache=False,
            classifier_dropout=None,
            vocab_size=None,
        )
        return config

    def forward(
        self,
        x: torch.Tensor,
        v_lengths: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
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

        attn_mask = (
            video_padding_mask.float()
            .masked_fill(video_padding_mask, float("-inf"))
            .unsqueeze(1)
            .unsqueeze(2)
        )

        # positional embedding
        position_ids = torch.arange(
            x.size(1), dtype=torch.long, device=x.device
        ).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)
        x += position_embeddings

        outputs = self.encoder(
            x, attention_mask=attn_mask, output_attentions=output_attentions
        )

        return outputs.last_hidden_state, v_lengths, video_padding_mask

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
    model = SignBertEncoder(
        hidden_size=1024,
        num_attention_heads=8,
        intermediate_size=2048,
        num_layers=6,
    ).cuda()
    x = torch.randn(2, 10, 1024).cuda()  # (batch_size, seq_len, hidden_size)
    v_lengths = torch.tensor([5, 8]).cuda()
    output, _, _ = model(x, v_lengths)
    print(output.shape)  # Should be (2, 30, 1024)
