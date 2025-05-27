import torch
from einops import rearrange, repeat
from torch import nn

from typing import Optional, Type
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.bert import BertConfig
from timm.models.vision_transformer import (
    Attention,
    DropPath,
    Mlp,
    LayerScale,
)


class SpatialTemporalAdapter(nn.Module):
    def __init__(
        self,
        hidden_size,
        target_hidden_size,
        num_heads,
        num_layers,
        num_extra_queries,
        max_length=512,
    ):
        super().__init__()
        self.num_extra_queries = num_extra_queries
        self.hidden_size = hidden_size
        self.target_hidden_size = target_hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.bert_config = self._get_config()
        self.bert_config.is_decoder = True

        self.position_embedding = nn.Embedding(max_length, hidden_size)

        self.extra_queries = nn.Parameter(
            torch.randn(1, 1, num_extra_queries, hidden_size), requires_grad=True
        )
        self.blocks = nn.ModuleList(
            [CrossAttentionQuery(self.bert_config) for _ in range(num_layers)]
        )

        self.cross_frame_blocks = nn.ModuleList(
            [CrossFrameAttention(self.bert_config) for _ in range(num_layers - 1)]
        )
        self.linear = nn.Linear(hidden_size, target_hidden_size)

    def _get_config(self):
        return BertConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            intermediate_size=None,
            num_hidden_layers=None,
            type_vocab_size=None,
            max_position_embeddings=None,
            pad_token_id=None,
            position_embedding_type=None,
            use_cache=None,
            classifier_dropout=None,
            vocab_size=None,
        )

    @staticmethod
    def length_to_mask(lengths, max_length=None):
        """
        Convert lengths to a binary mask.
        lengths: (B, T) tensor of lengths
        max_length: maximum length of the sequence
        """
        if max_length is None:
            max_length = lengths.max().item()
        mask = torch.arange(max_length, device=lengths.device).expand(
            lengths.size(0), max_length
        ) < lengths.unsqueeze(1)
        return mask.long()  # (B, max_length)

    def forward(self, x, v_length):
        # x: (B, T, HW, C)
        B, T, HW, C = x.shape

        extra_queries = repeat(self.extra_queries, "1 1 n c -> b t n c", b=B, t=T)

        # position embedding
        position_ids = torch.arange(T, device=x.device)
        position_embedding = self.position_embedding(position_ids)  # (T, C)
        position_embedding = rearrange(position_embedding, "t c -> 1 t 1 c")
        x = x + position_embedding  # (B, T, HW, C)

        cross_output = x
        temporal_attn_mask = self.length_to_mask(v_length, max_length=T)
        for id, block in enumerate(self.blocks):
            extra_queries = block(
                queries=extra_queries,
                visual_feats=cross_output,
            )
            if id == len(self.blocks) - 1:
                break

            cross_output, temporal_attn_mask = self.cross_frame_blocks[id](
                hidden_states=cross_output,
                attention_mask=temporal_attn_mask,
            )

        feats = extra_queries.mean(dim=-2)  # (B T C)
        feats = self.linear(feats)

        return (
            feats,
            temporal_attn_mask.sum(dim=1),  # new v_length
        )  # (B, T, num_extra_queries, target_hidden_size)


class CrossAttentionQuery(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.config.is_decoder = True
        self.attention = BertAttention(config)

    def forward(self, queries, visual_feats):
        """
        queries (B, T, n, C)
        visual_features: (B, T, hw, C)
        attention_mask: (B, T)
        """
        B, T, n, C = queries.shape

        queries = rearrange(queries, "b t n c -> (b t) n c")
        visual_feats = rearrange(visual_feats, "b t hw c -> (b t) hw c")

        output = self.attention(
            hidden_states=queries,
            encoder_hidden_states=visual_feats,
        )

        output = rearrange(
            output[0],
            "(b t) n c -> b t n c",
            b=B,
            t=T,
        )
        return output  # (B, T, n, C)


class CrossFrameAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.config.is_decoder = True
        self.attention = BertAttention(config)

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: (B, T, hw, C)
        attention_mask: (B, T)
        """
        B, T, HW, C = hidden_states.shape
        shifted_hidden_states, shifted_attention_mask = self.shift(
            hidden_states, shift_size=-1, attention_mask=attention_mask
        )

        hidden_states = rearrange(hidden_states, "b t hw c -> (b t) hw c")
        shifted_hidden_states = rearrange(
            shifted_hidden_states, "b t hw c -> (b t) hw c"
        )
        output = self.attention(
            hidden_states,
            encoder_hidden_states=shifted_hidden_states,
        )
        output = rearrange(
            output[0],
            "(b t) hw c -> b t hw c",
            b=B,
            t=T,
        )

        return output, shifted_attention_mask * attention_mask

    @staticmethod
    def shift(x, shift_size, attention_mask=None):
        """
        hidden_states: (B, T, hw, C)
        attention_mask: (B, T)
        """
        B, T, HW, C = x.shape

        shifted_x = torch.roll(x, shifts=shift_size, dims=1)
        if attention_mask is None:
            attention_mask = torch.ones([B, T], device=x.device)
        shifted_attention_mask = torch.roll(attention_mask, shifts=shift_size, dims=1)
        shifted_attention_mask[:, shift_size:] = (
            0  # Clear the first `shift_size` positions
        )

        return shifted_x, shifted_attention_mask


if __name__ == "__main__":
    # Example usage
    hidden_size = 768
    num_heads = 12
    num_layers = 2
    num_extra_queries = 10
    target_hidden_size = 512

    visual_adapter = SpatialTemporalAdapter(
        hidden_size, target_hidden_size, num_heads, num_layers, num_extra_queries
    )
    x = torch.randn(2, 30, 196, hidden_size)  # Example input
    v_length = torch.tensor([30, 25])  # Example video lengths for batch size 2
    output, v_length = visual_adapter(x, v_length)
    print(output.shape)  # Should be (2, 30, num_extra_queries, hidden_size)
