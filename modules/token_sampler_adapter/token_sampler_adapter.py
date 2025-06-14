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


class VisualSampleAdapter(nn.Module):
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
        max_length=512,
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
        self.linear = nn.Linear(hidden_size, target_hidden_size)
        self.positional_embedding = nn.Embedding(max_length, target_hidden_size)

    def forward(self, x, v_length):
        # x: (B, T, HW, C)
        B, T, HW, C = x.shape
        x = rearrange(x, "b t hw c -> (b t) hw c")

        extra_queries = repeat(self.extra_queries, "1 n c -> bt n c", bt=B * T)
        for block in self.blocks:
            extra_queries = block(extra_queries, x)

        extra_queries = rearrange(extra_queries, "(b t) n c -> b t n c", b=B, t=T)

        position_ids = (
            torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        )  # (B, T)
        position_embeddings = self.positional_embedding(
            position_ids
        )  # (B, T, Target_hidden_size)
        position_embeddings = repeat(
            position_embeddings, "b t c -> b t n c", n=self.num_extra_queries
        )
        feats = self.linear(extra_queries) + position_embeddings
        feats = rearrange(feats, "b t n c -> b (t n) c")  # (B, T, num_extra_queries, C)

        v_length = (
            v_length * self.num_extra_queries
        )  # Adjust v_length for extra queries

        return feats, v_length


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
    num_extra_queries = 4
    target_hidden_size = 1024

    visual_adapter = VisualSampleAdapter(
        hidden_size, target_hidden_size, num_heads, num_layers, num_extra_queries
    )
    x = torch.randn(2, 128, 196, hidden_size)  # Example input
    output = visual_adapter(x, v_length=torch.tensor([128, 128]))  # Example v_length
    print(output[0].shape)  # Should be (2, 30, num_extra_queries, hidden_size)
