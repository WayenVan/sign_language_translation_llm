import torch
import torch.nn.functional as F
from geomloss import SamplesLoss


def masked_emd_batch(Fs, Fq, mask_s, mask_q, blur=0.05):
    """
    计算批次中的带掩码 Earth Mover's Distance（可导）

    参数：
    - Fs: 支持集特征，形状为 [B, Ns, C]
    - Fq: 查询集特征，形状为 [B, Nq, C]
    - mask_s: 支持集掩码，形状为 [B, Ns]（0 或 1）
    - mask_q: 查询集掩码，形状为 [B, Nq]（0 或 1）
    - epsilon: Sinkhorn 正则化系数

    返回：
    - emd_costs: 形状 [B] 的 EMD 值，支持反向传播
    """
    # 归一化特征向量
    Fs_n = F.normalize(Fs, p=2, dim=2, eps=1e-12)  # [B, Ns, C]
    Fq_n = F.normalize(Fq, p=2, dim=2, eps=1e-12)  # [B, Nq, C]

    # 将掩码转为权重，并标准化
    a = mask_s.float()
    a = a / (a.sum(dim=1, keepdim=True) + 1e-12)  # [B, Ns]
    b = mask_q.float()
    b = b / (b.sum(dim=1, keepdim=True) + 1e-12)  # [B, Nq]

    # 可导 Sinkhorn 距离
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur)
    emd_costs = loss_fn(a, Fs_n, b, Fq_n)  # 输出 [B]，支持 backprop

    return emd_costs.mean()


if __name__ == "__main__":
    # 测试代码
    B, Ns, Nq, C = 2, 5, 6, 3
    Fs = torch.randn(B, Ns, C)
    Fq = torch.randn(B, Nq, C)
    mask_s = torch.randint(0, 2, (B, Ns))
    mask_q = torch.randint(0, 2, (B, Nq))

    emd_costs = masked_emd_batch(Fs, Fq, mask_s, mask_q)
    print("EMD Costs:", emd_costs)
