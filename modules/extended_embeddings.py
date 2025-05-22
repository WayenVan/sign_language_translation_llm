import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class CustomEmbeddingLayer(nn.Module):
    def __init__(
        self,
        old_vocab_size,
        new_vocab_size,
        embedding_dim,
        padding_idx,
        pretrained_weights,
    ):
        super().__init__()
        # Embedding for old (pre-trained) tokens - frozen
        self.old_embeddings = nn.Embedding(old_vocab_size, embedding_dim, padding_idx)
        self.old_embeddings.weight.data.copy_(
            pretrained_weights
        )  # Load pre-trained weights
        self.old_embeddings.weight.requires_grad = False  # Freeze old embeddings

        # Embedding for new tokens - trainable
        self.new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
        # new_embeddings.weight is randomly initialized by default, requires_grad=True

        self.old_vocab_size = old_vocab_size

    def forward(self, input_ids):
        # input_ids: tensor of token IDs
        # Split input_ids into old and new token IDs
        is_old_token = input_ids < self.old_vocab_size
        old_token_ids = input_ids[is_old_token]
        new_token_ids = (
            input_ids[~is_old_token] - self.old_vocab_size
        )  # Adjust indices for new_embeddings

        # Look up embeddings
        old_embeds = self.old_embeddings(old_token_ids)
        new_embeds = self.new_embeddings(new_token_ids)

        # Reconstruct the output embeddings in original order
        # This part can be tricky and requires careful indexing/scatter
        # A simpler approach for batch processing might be to process them separately
        # and then concatenate/reconstruct.
        # For simplicity, let's assume we return them as separate tensors or rely on batching.
        # A more robust way might involve a custom scatter or creating a zero tensor and filling.

        # For a practical approach, it's often easier to build a combined embedding matrix initially,
        # then freeze parts of it, as discussed in the next option (though it has its own caveats).
        # Or, just accept that the whole embedding layer is trainable, but new tokens are the primary drivers.

        # More robust (but might be less efficient for very large batches with mixed types):
        output_embeds = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.old_embeddings.embedding_dim,
            device=input_ids.device,
        )
        output_embeds[is_old_token] = self.old_embeddings(input_ids[is_old_token])
        output_embeds[~is_old_token] = self.new_embeddings(
            input_ids[~is_old_token] - self.old_vocab_size
        )
        return output_embeds
