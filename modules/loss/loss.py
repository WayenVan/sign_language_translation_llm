from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    """
    Loss function for the model.
    """

    def __init__(self, padding_idx):
        super(Loss, self).__init__()
        self.padding_idx = padding_idx

    def forward(
        self, output, target_token_llm_features, target_tokens, target_token_mask
    ):
        """
        @param output: The output from the decoder, namedtuple containing:
            - logits: The logits from the decoder [batch_size, seq_len, vocab_size]
            - token_llm_features: The token features from the LLM [batch_size, seq_len, llm_feature_dim]
        @param target_token_llm_features: The target token features from the LLM [batch_size, seq_len, llm_feature_dim]
        @param target_tokens: The idx of target tokens [batch_size, seq_len]
        @param target_token_mask: The mask for the target tokens [batch_size, seq_len]
        """

        loss = 0.0
        ce_loss = None
        mse_loss = None

        prob = F.log_softmax(output.logits, dim=-1)
        ce_loss = F.nll_loss(
            prob.flatten(0, 1),
            target_tokens.flatten(),
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        loss += ce_loss

        mse_loss = F.mse_loss(
            output.token_llm_features,
            target_token_llm_features,
            reduction="mean",
        )
        loss += mse_loss
        return self.LossOutputs(
            loss=loss,
            ce_loss=ce_loss,
            mse_loss=mse_loss,
        )

    LossOutputs = namedtuple(
        "LossOutputs",
        [
            "loss",
            "ce_loss",
            "mse_loss",
        ],
    )
