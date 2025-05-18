import torch
import torch.nn.functional as F


def masked_cross_sequence_contrastive_loss(
    source, target, source_lens, target_lens, temperature=0.07
):
    """
    source:       [B, Ls, D]
    target:       [B, Lt, D]
    source_lens:  [B]   (有效 token 数)
    target_lens:  [B]   (有效 token 数)
    """
    B, Ls, D = source.shape
    _, Lt, _ = target.shape

    # Normalize
    source = F.normalize(source, dim=-1)  # [B, Ls, D]
    target = F.normalize(target, dim=-1)  # [B, Lt, D]

    # Similarity: [B, Ls, Lt]
    sim_matrix = (
        torch.matmul(source, target.transpose(1, 2)) / temperature
    )  # [B, Ls, Lt]

    # Mask for valid source positions
    src_mask = torch.arange(Ls, device=source.device).unsqueeze(
        0
    ) < source_lens.unsqueeze(1)  # [B, Ls]
    tgt_mask = torch.arange(Lt, device=target.device).unsqueeze(
        0
    ) < target_lens.unsqueeze(1)  # [B, Lt]

    # Expand to [B, Ls, Lt] for masking
    src_mask_exp = src_mask.unsqueeze(2)  # [B, Ls, 1]
    tgt_mask_exp = tgt_mask.unsqueeze(1)  # [B, 1, Lt]
    valid_mask = src_mask_exp & tgt_mask_exp  # [B, Ls, Lt]

    # Set invalid positions in similarity to -inf so they are ignored in argmax & softmax
    sim_matrix = sim_matrix.masked_fill(~valid_mask, -float("inf"))  # [B, Ls, Lt]

    # Flatten for cross-entropy: keep only valid source positions
    flat_logits = sim_matrix.view(-1, Lt)  # [B*Ls, Lt]
    flat_mask = src_mask.view(-1)  # [B*Ls]
    valid_indices = flat_mask.nonzero(as_tuple=True)[0]  # valid source positions only
    logits = flat_logits[valid_indices]  # [N_valid_src, Lt]

    # For each valid source token, find most similar valid target token
    argmax_idx = torch.argmax(sim_matrix, dim=2)  # [B, Ls]
    argmax_idx = argmax_idx.view(-1)[valid_indices]  # [N_valid_src]

    # Final loss
    loss = F.cross_entropy(logits, argmax_idx)
    return loss


def masked_bi_directional_contrastive_loss(
    source, target, source_lens, target_lens, temperature=0.07
):
    """
    source:       [B, Ls, D]
    target:       [B, Lt, D]
    source_lens:  [B]
    target_lens:  [B]
    """
    B, Ls, D = source.shape
    _, Lt, _ = target.shape

    source = F.normalize(source, dim=-1)
    target = F.normalize(target, dim=-1)

    # 相似度矩阵 source->target 和 target->source
    sim_st = torch.matmul(source, target.transpose(1, 2)) / temperature  # [B, Ls, Lt]
    sim_ts = torch.matmul(target, source.transpose(1, 2)) / temperature  # [B, Lt, Ls]

    # mask 同之前定义
    src_mask = torch.arange(Ls, device=source.device).unsqueeze(
        0
    ) < source_lens.unsqueeze(1)  # [B, Ls]
    tgt_mask = torch.arange(Lt, device=target.device).unsqueeze(
        0
    ) < target_lens.unsqueeze(1)  # [B, Lt]

    src_mask_exp = src_mask.unsqueeze(2)  # [B, Ls, 1]
    tgt_mask_exp = tgt_mask.unsqueeze(1)  # [B, 1, Lt]
    valid_mask_st = src_mask_exp & tgt_mask_exp  # [B, Ls, Lt]
    valid_mask_ts = tgt_mask.unsqueeze(2) & src_mask.unsqueeze(1)  # [B, Lt, Ls]

    sim_st = sim_st.masked_fill(~valid_mask_st, -float("inf"))
    sim_ts = sim_ts.masked_fill(~valid_mask_ts, -float("inf"))

    # flatten + filter valid positions
    logits_st = sim_st.view(-1, Lt)
    logits_ts = sim_ts.view(-1, Ls)

    src_mask_flat = src_mask.view(-1)
    tgt_mask_flat = tgt_mask.view(-1)

    valid_st = src_mask_flat.nonzero(as_tuple=True)[0]
    valid_ts = tgt_mask_flat.nonzero(as_tuple=True)[0]

    logits_st = logits_st[valid_st]  # [N_valid_src, Lt]
    logits_ts = logits_ts[valid_ts]  # [N_valid_tgt, Ls]

    # 正样本索引
    pos_idx_st = torch.argmax(sim_st, dim=2).view(-1)[valid_st]
    pos_idx_ts = torch.argmax(sim_ts, dim=2).view(-1)[valid_ts]

    loss_st = F.cross_entropy(logits_st, pos_idx_st)
    loss_ts = F.cross_entropy(logits_ts, pos_idx_ts)

    return (loss_st + loss_ts) / 2


if __name__ == "__main__":
    # Example usage
    source_seq = torch.randn(2, 5, 128)  # 32 samples, 128-dimensional embeddings
    target_seq = torch.randn(2, 8, 128)
    source_len = torch.tensor([3, 4])
    target_len = torch.tensor([4, 6])

    loss = masked_cross_sequence_contrastive_loss(
        source_seq, target_seq, source_len, target_len
    )
    print(f"Contrastive Loss: {loss.item()}")
