from tkinter import N
import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Tuple


from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    Unpack,
    Cache,
    FlashAttentionKwargs,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import DynamicCache


class LlamaCrossDecoderLayer(nn.Module):
    def __init__(self, llama_config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = llama_config.hidden_size

        self.self_attn = LlamaAttention(config=llama_config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(llama_config)
        self.input_layernorm = LlamaRMSNorm(
            llama_config.hidden_size, eps=llama_config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            llama_config.hidden_size, eps=llama_config.rms_norm_eps
        )
        self.post_cross_attention_layernorm = LlamaRMSNorm(
            llama_config.hidden_size, eps=llama_config.rms_norm_eps
        )

        self.cross_attention = nn.MultiheadAttention(
            llama_config.hidden_size,
            llama_config.num_attention_heads,
            llama_config.attention_dropout,
            llama_config.attention_bias,
            batch_first=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_hidden_states: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        visual_padding_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=self_attn_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states, cross_atten_weight = self.cross_attention(
            hidden_states,
            visual_hidden_states,
            visual_hidden_states,
            key_padding_mask=visual_padding_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_cross_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_atten_weight)

        return outputs


if __name__ == "__main__":
    config = LlamaConfig(
        hidden_size=1024,
        intermediate_size=2048,  # in feed forward hidden_size
        num_attention_heads=8,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        pretraining_tp=None,
        head_dim=None,
        vocab_size=None,
        use_cache=None,
        pad_token_id=None,
        num_hidden_layers=None,
        bos_token_id=None,
        eos_token_id=None,
    )
    model = LlamaCrossDecoderLayer(config, 0).cuda()
    rotray = LlamaRotaryEmbedding(config, device="cuda")
    hidden_states = torch.randn(2, 1, 1024).cuda()
    visual_hidden_states = torch.randn(2, 1, 1024).cuda()
    position_ids = torch.arange(0, 4).unsqueeze(0).expand(2, -1).cuda()
    position_embeddings = rotray(hidden_states, position_ids=position_ids)
    output = model(
        hidden_states, visual_hidden_states, position_embeddings=position_embeddings
    )
    for name, param in model.named_parameters():
        print(name, param.shape)

    cache = DynamicCache()
    for i in range(4):
        output = model(
            hidden_states,
            visual_hidden_states,
            position_embeddings=position_embeddings,
            past_key_value=cache,
            use_cache=True,
        )

    print(output)
