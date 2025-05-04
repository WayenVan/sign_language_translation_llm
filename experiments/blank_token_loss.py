import numpy as np
import torch
from torch.nn import functional as F
import random


def blank_token_loss(outputs, targets, reduction="none"):
    """
    @param outputs: model outputs, namedtuple containing:
        - queries: model logits, shape (batch_size, queries) binary logits
    @targets: targets, a list contains the non-empty indexs of queries:
        - valid_queries: list of binary mask (no_empty_indexes), len(valid_queries) = batch_size
    """
    queries = outputs.queries
    valid_queries = targets.valid_queries

    B, Q = queries.shape

    assert B == len(valid_queries), f"Batch size mismatch: {B} != {len(targets)}"

    labels = torch.zeros((B, Q), device=queries.device)
    b_indx = torch.arange(B, device=queries.device).repeat_interleave(
        torch.tensor([len(v) for v in valid_queries], device=queries.device)
    )
    q_indx = torch.cat(valid_queries, dim=0)
    labels[b_indx, q_indx] = 1

    # calculate the positive weight
    positives = labels.sum()
    total = len(labels.flatten())
    pos_weight = (total - positives) / total

    return F.binary_cross_entropy_with_logits(
        queries, labels, reduction=reduction, pos_weight=pos_weight
    )


if __name__ == "__main__":
    # Example usage
    B, Q = 4, 5
    queries = torch.randn(B, Q).cuda()

    valid_queries = [
        torch.randint(0, Q, (random.randint(1, Q),)).cuda() for _ in range(B)
    ]
    targets = type("Targets", (object,), {"valid_queries": valid_queries})
    outputs = type("Outputs", (object,), {"queries": queries})

    loss = blank_token_loss(outputs, targets)
    print("Loss:", loss.item())
