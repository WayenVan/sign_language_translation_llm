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
import copy
import torch.nn.functional as F


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class TokenSampleTConvDapter(nn.Module):
    def __init__(
        self,
        hidden_size,
        target_hidden_size,
        num_heads,
        num_layers,
        num_extra_queries,
        t_conv_type=2,
        mlp_depth=1,
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
        self.token_sampler = nn.ModuleList(
            [
                ViTAttentionBlock(
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
        self.mlp1 = build_mlp(
            mlp_depth, hidden_size * self.num_extra_queries, hidden_size
        )
        self.temporal_conv = TemporalConv(
            hidden_size, hidden_size, conv_type=t_conv_type
        )
        self.mlp2 = build_mlp(mlp_depth, hidden_size, target_hidden_size)

        self.positional_embedding = nn.Embedding(max_length, target_hidden_size)

    def forward(self, x, v_length):
        # x: (B, T, HW, C)
        B, T, HW, C = x.shape
        x = rearrange(x, "b t hw c -> (b t) hw c")

        extra_queries = repeat(self.extra_queries, "1 n c -> bt n c", bt=B * T)
        for block in self.token_sampler:
            extra_queries = block(extra_queries, x)

        extra_queries = rearrange(
            extra_queries, "(b t) n c -> b t (n c)", b=B, t=T
        )  # (B, T, num_extra_queries * hidden_size)
        feats = self.mlp1(extra_queries)  # (B, T, hidden_size)
        feats = rearrange(feats, "b t c -> b c t")  # (B, hidden_size, T)
        feats, _, v_length = self.temporal_conv(feats, v_length)  # (B, hidden_size, T)

        position_ids = (
            torch.arange(feats.shape[1], device=x.device).unsqueeze(0).expand(B, -1)
        )  # (B, T)
        position_embeddings = self.positional_embedding(
            position_ids
        )  # (B, T, Target_hidden_size)

        feats = self.mlp2(feats) + position_embeddings  # (B, T, Target_hidden_size)

        return feats, v_length


class ViTAttentionBlock(nn.Module):
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


class TemporalConv(nn.Module):
    def __init__(
        self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1
    ):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ["K3"]
        elif self.conv_type == 1:
            self.kernel_size = ["K5", "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ["K5", "P2", "K5", "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == "P":
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == "K":
                modules.append(
                    nn.Conv1d(
                        input_sz,
                        self.hidden_size,
                        kernel_size=int(ks[1]),
                        stride=1,
                        padding=0,
                    )
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        # 直接遍历所有内核操作并更新长度
        for ks in self.kernel_size:
            if ks[0] == "P":
                # 池化：长度减半（向下取整）
                lgt = torch.div(lgt, int(ks[1]), rounding_mode="floor")
            elif ks[0] == "K":
                # 卷积：长度减少 (kernel_size - 1)
                lgt -= int(ks[1]) - 1
        return lgt

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = (
            None
            if self.num_classes == -1
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        )
        visual_feat = visual_feat.permute(0, 2, 1)  # (B, hidden_size, T)
        if logits is not None:
            logits = logits.permute(0, 2, 1)
        return visual_feat, logits, lgt


if __name__ == "__main__":
    # Example usage
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    num_extra_queries = 4
    target_hidden_size = 1024

    visual_adapter = TokenSampleTConvDapter(
        hidden_size, target_hidden_size, num_heads, num_layers, num_extra_queries
    )
    x = torch.randn(2, 128, 196, hidden_size)  # Example input
    output = visual_adapter(x, v_length=torch.tensor([128, 127]))  # Example v_length
    print(output[0].shape)  # Should be (2, 30, num_extra_queries, hidden_size)
