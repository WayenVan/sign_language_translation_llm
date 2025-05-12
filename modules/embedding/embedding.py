import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple


class LLMCompressEmbedding(nn.Module):
    def __init__(self, num_idx, hidden_states, llm_hidden_states, dropout) -> None:
        super().__init__()
        self.num_idx = num_idx
        self.llm_hidden_states = llm_hidden_states
        self.hidden_states = hidden_states

        self.embedding = nn.Embedding(num_idx, llm_hidden_states)
        self.compress = nn.Linear(llm_hidden_states, hidden_states)
        self.dropout = nn.Dropout(dropout)

    LLMCompressEmbeddingOutputs = namedtuple(
        "LLMCompressEmbeddingOutputs", ["embeddings", "llm_embeddings"]
    )

    def forward(self, x):
        """
        @param x : (batch_size, seq_len)
        """
        x = self.embedding(x)
        llm_hidden_states = x
        x = self.compress(x)
        x = self.dropout(x)
        return self.LLMCompressEmbeddingOutputs(
            embeddings=x, llm_embeddings=llm_hidden_states
        )
