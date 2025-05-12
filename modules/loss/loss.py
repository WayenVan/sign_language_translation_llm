from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange


class Loss(nn.Module):
    """
    Loss function for the model.
    """

    def __init__(self, ce_weight, mse_weight, cross_model_weight):
        super(Loss, self).__init__()
        self.ce_weight = ce_weight
        self.mse_weight = mse_weight
        self.cross_model_weight = cross_model_weight

    def forward(
        self,
        pred_decoder_logits,
        pred_textual_features,
        pred_video_features,
        pred_token_llm_features,
        textual_length,
        video_length,
        target_token_llm_features,
        target_decoder_tokens,
        padding_idx,
    ):
        """
        @param pred_decoder_logits: The logits from the decoder [batch_size, seq_len, vocab_size]
        @param pred_textual_features: The textual features from shared encoder [batch_size, seq_len-1, feature_dim]
        @param pred_video_features: The video features from shared encoder [batch_size,video_seq_len, feature_dim]
        @param pred_token_llm_features: The token features from the LLM [batch_size, seq_len, llm_feature_dim]
        @param textual_length: The length of the textual features [batch_size]
        @param video_length: The length of the video features [batch_size]
        @param target_token_llm_features: The target token features from the LLM [batch_size, seq_len, llm_feature_dim]
        @param target_tokens: The idx of target tokens [batch_size, seq_len]
        """

        loss = 0.0
        ce_loss = 0.0
        mse_loss = 0.0
        cross_modal_loss = 0.0

        if self.ce_weight > 0:
            prob = F.log_softmax(pred_decoder_logits, dim=-1)
            ce_loss = F.nll_loss(
                prob.flatten(0, 1),
                target_decoder_tokens.flatten(0, 1),
                ignore_index=padding_idx,
                reduction="mean",
            )
            loss += ce_loss

        if self.mse_weight > 0:
            mse_loss = F.mse_loss(
                pred_token_llm_features,
                target_token_llm_features,
                reduction="mean",
            )
            loss += mse_loss

        if self.cross_model_weight > 0:
            # Cross-modal attention loss
            cross_modal_loss = cross_modal_attention(
                pred_video_features,
                pred_textual_features,
                video_length,
                textual_length,
            )
            loss += self.cross_model_weight * cross_modal_loss

        return self.LossOutputs(
            loss=loss,
            ce_loss=ce_loss * self.ce_weight,
            mse_loss=mse_loss * self.mse_weight,
            cross_modal_loss=cross_modal_loss * self.cross_model_weight,
        )

    LossOutputs = namedtuple(
        "LossOutputs",
        [
            "loss",
            "ce_loss",
            "mse_loss",
            "cross_modal_loss",
        ],
    )


def create_binary_mask(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max()
    mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]
    return ~mask


def binary_mask_to_additive_mask(mask):
    mask = ~mask
    return mask.float().masked_fill(mask, float("-inf"))  # [B, T]


def masked_mse_loss(predictions, targets, mask=None):
    """
    predictions: [B, T, D] or [B, T]
    targets: same shape as predictions
    mask: [B, T] where 1=valid, 0=padding
    """
    if mask is None:
        return F.mse_loss(predictions, targets, reduction="mean")
    squared_error = (predictions - targets) ** 2
    # Apply mask and average only over valid elements
    squared_error = squared_error * mask.unsqueeze(-1)
    squared_error = squared_error.sum() / mask.sum()
    return squared_error


def cross_modal_attention(frame_embs, word_embs, frame_length=None, word_length=None):
    # frame_embs: [B, T, D], word_embs: [B, N, D], frame_legnth [B], word_lenth [B]
    sim_matrix = torch.matmul(frame_embs, word_embs.transpose(-1, -2))  # [B, T, N]

    # Text-to-Frame Attention
    word_length_mask = None
    if frame_length is not None:
        frame_length_mask = create_binary_mask(frame_length, frame_embs.shape[1])
        frame_length_mask_add = binary_mask_to_additive_mask(frame_length_mask)
    attn_word_to_frame = F.softmax(
        sim_matrix + rearrange(frame_length_mask_add, "b t -> b t 1"), dim=1
    )  # [B, T, N]
    attended_frame_feats = torch.matmul(
        attn_word_to_frame.transpose(-1, -2), frame_embs
    )  # [B, N, D]

    # Frame-to-Text Attention
    word_length_mask = None
    if word_length is not None:
        word_length_mask = create_binary_mask(word_length, word_embs.shape[1])  # [B, N]
        word_length_mask_add = binary_mask_to_additive_mask(word_length_mask)
    attn_frame_to_word = F.softmax(
        sim_matrix + rearrange(word_length_mask_add, "b n -> b 1 n"), dim=2
    )  # [B, T, N]
    attended_word_feats = torch.matmul(attn_frame_to_word, word_embs)  # [B, T, D]

    loss = masked_mse_loss(
        attended_frame_feats, word_embs.detach(), word_length_mask
    ) + masked_mse_loss(attended_word_feats, frame_embs.detach(), frame_length_mask)

    return loss


if __name__ == "__main__":
    cross_modal_attention(
        torch.randn(2, 5, 768),
        torch.randn(2, 10, 768),
        torch.tensor([5, 3]),
        torch.tensor([10, 8]),
    )
