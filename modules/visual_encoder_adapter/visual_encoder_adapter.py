import torch
from einops import rearrange, repeat
from torch import nn

from typing import Optional, Type
from timm.models.vision_transformer import (
    Attention,
    DropPath,
    Mlp,
    LayerScale,
)
from .tconv import TemporalConv1D


class VisualAdapter(nn.Module):
    def __init__(
        self,
        hidden_size,
        target_hidden_size,
        num_heads,
        num_layers,
        num_extra_queries,
        mlp_ratio=2.0,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        max_token_length: Optional[int] = 512,
    ):
        super().__init__()
        self.num_extra_queries = num_extra_queries
        self.extra_queries = nn.Parameter(
            torch.randn(1, num_extra_queries, hidden_size), requires_grad=True
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    mlp_layer=Mlp,
                )
                for _ in range(num_layers)
            ]
        )
        self.position_embedding = nn.Embedding(max_token_length, hidden_size)
        self.linear = nn.Linear(hidden_size, target_hidden_size)
        self.aggregation = Aggregation(hidden_size)

    def forward(self, x, t_length):
        # x: (B, T, HW, C)
        B, T, HW, C = x.shape
        x = rearrange(x, "b t hw c -> (b t) hw c")

        extra_queries = repeat(self.extra_queries, "1 n c -> bt n c", bt=B * T)
        for block in self.blocks:
            extra_queries = block(extra_queries, x)

        extra_queries = rearrange(extra_queries, "(b t) n c -> b t n c", b=B, t=T)

        # Add positional embeddings
        position_ids = torch.arange(T, device=x.device)
        position_embeddings = rearrange(
            self.position_embedding(position_ids), "t c -> 1 t 1 c"
        )
        extra_queries = extra_queries + position_embeddings

        # aggregate extra queries
        extra_queries, t_length = self.aggregation(extra_queries, t_length)

        # flatten
        t_length = t_length * self.num_extra_queries
        feats = rearrange(extra_queries, "b t n c -> b (t n) c")
        feats = self.linear(feats)

        return feats, t_length


class Aggregation(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conv = TemporalConv1D(
            input_size=hidden_size,
            out_size=hidden_size,
            bottleneck_size=hidden_size,
            conv_type=["K3", "P2", "K3", "P2"],
            dropout=0.0,
        )

    def forward(self, x, t_length):
        # x: (B, T, N,  C)
        B, T, N, C = x.shape
        x = rearrange(x, "b t n c -> (b n) c t")
        x, t_length = self.conv(x, t_length)
        x = rearrange(x, "(b n) c t -> b t n c", b=B, n=N)
        return x, t_length


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm0 = norm_layer(dim)
        self.norm1 = norm_layer(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=True,
            batch_first=True,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        keys = self.norm0(keys)
        x = queries + self.drop_path1(
            self.ls1(self.attn(self.norm1(queries), keys, keys)[0])
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


if __name__ == "__main__":
    # Example usage
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    num_extra_queries = 10

    visual_adapter = VisualAdapter(
        hidden_size, num_heads, num_layers, num_extra_queries
    )
    x = torch.randn(2, 30, 196, hidden_size)  # Example input
    output = visual_adapter(x)
    print(output.shape)  # Should be (2, 30, num_extra_queries, hidden_size)
