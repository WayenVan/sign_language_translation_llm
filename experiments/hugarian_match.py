import torch
from scipy.optimize import linear_sum_assignment
from einops import rearrange
from torch import nn
from collections import namedtuple


class HungarianMatcher(nn.Module):
    def __init__(self, weight_dist, weight_no_empty, weight_order):
        super().__init__()
        self.weight_dist = weight_dist
        self.weight_no_empty = weight_no_empty
        self.weight_order = weight_order

    @torch.no_grad()
    def forward(self, outputs, targets, p=1):
        """
        reference to the detr github repo
        @Params outputs: The namedtuple of the model output, as least have those entries
            "pred_logits": Tensor of dim [batch_size, num_queries, keyword_embeddings] with the classification logits
            "pred_no_empty": Tensor of dim [batch_size, num_queries] with the logits of no empty
            "pred_order_norm": Tensor of dim [batch_size, num_queries] the predicted normalization order logits before sigmoid
        @Params targets: The namedtuple of the model target, as least have those entries
            "embeddings": Tensor of dim [(batch_size, num_targets), keywords_embeddings] concated keywords batch,
            for num_targets is variable in different batches
            "target_length": Tensor of dim [batch_size] , contains the length of num_targets in different batches
        @Params p: The p of the p-norm for calculate the distance between outputs and targets, default is 2
        @Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        pred_logits = outputs.pred_logits
        pred_no_empty = outputs.pred_no_empty
        pred_order_norm = outputs.pred_order_norm
        target_embeddings = targets.embeddings
        target_length = targets.target_length
        device = pred_logits.device

        B, Q, D = pred_logits.shape

        assert D == target_embeddings.shape[-1], (
            f"pred_logits and target_embeddings should have the same embedding dimension, but got {D} and {target_embeddings.shape[-1]} respectively."
        )

        pred_logits = rearrange(pred_logits, "b q d -> (b q) d")
        pred_no_empty = rearrange(pred_no_empty, "b q -> (b q)")

        cost_no_empty = -torch.log(torch.sigmoid(pred_no_empty)).unsqueeze(-1)
        cost_dist = torch.cdist(pred_logits, target_embeddings, p=p)

        # cost of order
        max_length = target_length.max().item()
        base = torch.arange(max_length, device=device).unsqueeze(0)
        base = base.expand(B, -1)
        normalized_order = base / (target_length.unsqueeze(1) - 1.0)
        mask = base < target_length.unsqueeze(1)
        mask = rearrange(mask, "b qm -> (b qm)")
        normalized_order = rearrange(normalized_order, "b qm -> (b qm)")
        normalized_order = normalized_order[mask]
        pred_order_norm = rearrange(pred_order_norm, "b q -> (b q)").sigmoid()

        cost_order = torch.cdist(
            pred_order_norm.unsqueeze(-1), normalized_order.unsqueeze(-1), p=p
        )

        C = (
            cost_no_empty * self.weight_no_empty
            + cost_dist * self.weight_dist
            + cost_order * self.weight_order
        )
        # [(b q), bt]
        C = rearrange(C, "(b q) bt -> b q bt", b=B, q=Q).cpu()

        sizes = [lgth.item() for lgth in target_length]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


if __name__ == "__main__":
    pred_logits = torch.rand(2, 100, 1024).cuda()
    pred_no_empty = torch.rand(2, 100).cuda()
    pred_norm_order = torch.rand(2, 100).cuda()
    target_embeddings = torch.rand(10, 1024).cuda()
    target_length = torch.tensor([4, 6], dtype=torch.int64).cuda()

    targets = type("Targets", (object,), {})()
    targets.embeddings = target_embeddings
    targets.target_length = target_length
    outputs = type("Outputs", (object,), {})()
    outputs.pred_logits = pred_logits
    outputs.pred_no_empty = pred_no_empty
    outputs.pred_order_norm = pred_norm_order

    hungarian_matcher = HungarianMatcher(
        weight_dist=1.0, weight_no_empty=1.0, weight_order=1.0
    )
    indices = hungarian_matcher(outputs, targets)
    print(indices)
