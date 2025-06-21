import numpy as np
import torch


def length_to_mask(lengths, max_length=None):
    """
    Convert lengths to a boolean mask.
    lengths: [B]
    max_length: int, optional
    """
    if max_length is None:
        max_length = lengths.max().item()
    B = lengths.size(0)
    mask = torch.arange(max_length, device=lengths.device).expand(
        B, max_length
    ) < lengths.unsqueeze(1)
    return mask.long()  # (B, max_length)


def contrastive_loss(video, video_length, text, text_mask):
    """
    Compute contrastive loss between video and text embeddings.
    Args:
        video: Video embeddings of shape (B, T, D)
        video_length: Lengths of video sequences of shape (B,)
        text: Text embeddings of shape (B, L, D)
        text_mask: Attention mask for text of shape (B, L)
    """

    video_feats = torch.nn.functional.normalize(video, dim=-1, p=2)
    text_feats = torch.nn.functional.normalize(text, dim=-1, p=2).detach()
    video_mask = length_to_mask(video_length, max_length=video_feats.size(1))  # (B, T)
    padding_mask = text_mask.unsqueeze(1) * video_mask.unsqueeze(2)  # (B, T, L)
    padding_mask = padding_mask.float()  # Convert to float for masking
    padding_mask = padding_mask.masked_fill(
        padding_mask == 0, float("-inf")
    )  # Set padding to -inf
    padding_mask = padding_mask.masked_fill(
        padding_mask == 1, 0.0
    )  # Set non-padding to 0.0

    similarity = torch.bmm(video_feats, text_feats.transpose(1, 2))  # (B, T, L)
    similarity = similarity + padding_mask  # Apply padding mask
    similarity = torch.log_softmax(similarity, dim=-1)  # Log softmax

    value, indeces = torch.topk(similarity, k=5, dim=-2)  # Get top-k indices, (B, 5, L)
    mask = torch.zeros_like(similarity, dtype=torch.float)  # (B, T, L)
    mask.scatter_(-2, indeces, 1.0)  # Set top-k indices to 1

    loss = -(similarity * mask).sum(dim=-1).mean()  # Compute loss
    return loss


if __name__ == "__main__":
    # Example usage
    device = "cuda"
    B, T, D = 2, 10, 512  # Batch size, video length, feature dimension
    video = torch.randn(B, T, D).to(device)
    video_length = torch.tensor([10, 8]).to(device)  # Video lengths
    L = 20  # Text length
    text = torch.randn(B, L, D).to(device)
    text_mask = torch.ones(B, L).to(device).long()  # Text attention mask
    text_mask[:, 15:] = 0  # Simulate some padding in the text
    loss = contrastive_loss(video, video_length, text, text_mask)
