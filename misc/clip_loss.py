import torch
import torch.nn.functional as F
import math


def clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor = None,
):
    """
    Compute CLIP-style contrastive loss.
    Args:
        image_features: Tensor of shape [B, N], L2-normalized or will be normalized.
        text_features: Tensor of shape [B, N], same as above.
        logit_scale: Optional scalar tensor or float. If None, default to 1 / 0.07 ≈ 14.2857.
    Returns:
        Scalar loss value.
    """

    # L2 normalize features
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)

    # Similarity logits
    logits_per_image = image_features @ text_features.T  # [B, B]
    logits_per_text = logits_per_image.T  # [B, B]

    # Default logit scale if not provided
    if logit_scale is None:
        logit_scale = torch.tensor(1 / 0.07).to(image_features.device)
    elif isinstance(logit_scale, float):
        logit_scale = torch.tensor(logit_scale).to(image_features.device)

    # Clamp logit_scale to prevent numerical overflow
    logit_scale.clamp(min=math.log(1), max=math.log(100)).exp()

    logits_per_image = logits_per_image * logit_scale
    logits_per_text = logits_per_text * logit_scale

    # Ground truth indices
    ground_truth = torch.arange(image_features.size(0), device=image_features.device)

    # CrossEntropyLoss for both directions
    loss_i2t = F.cross_entropy(logits_per_image, ground_truth)
    loss_t2i = F.cross_entropy(logits_per_text, ground_truth)

    return (loss_i2t + loss_t2i) / 2

