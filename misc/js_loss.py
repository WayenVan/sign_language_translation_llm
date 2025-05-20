import torch
import torch.nn.functional as F


def js_inverted_loss_from_log_probs(log_p, log_q, mask=None, base=1.0, eps=1e-8):
    """
    log_p, log_q: [B, C] log_softmax 输出
    mask: [B]，LongTensor 或 BoolTensor，1 表示有效，0 表示无效，或者 None（不做mask）
    base: 反转基准，建议 >= log(2)
    """
    if mask is not None:
        mask = mask.bool()
        if mask.sum() == 0:
            # 全部无效时返回0
            return torch.tensor(0.0, device=log_p.device, dtype=log_p.dtype)
        log_p = log_p[mask]
        log_q = log_q[mask]

    p = log_p.exp() + eps
    q = log_q.exp() + eps
    m = 0.5 * (p + q)

    kl_pm = F.kl_div(log_p, m, reduction="batchmean", log_target=False)
    kl_qm = F.kl_div(log_q, m, reduction="batchmean", log_target=False)

    js = 0.5 * (kl_pm + kl_qm)
    return base - js


if __name__ == "__main__":
    # Example usage
    log_p = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    log_q = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    mask = torch.tensor([1, 1])
    base = 1.0
    eps = 1e-8

    loss = js_inverted_loss_from_log_probs(log_p, log_q, mask, base, eps)
    print(loss)
