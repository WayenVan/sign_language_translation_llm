import torch
from typing import Optional


class CircularQueue:
    def __init__(
        self,
        max_size: int,
        feature_dim: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        高效环形队列，支持MoCo等对比学习任务

        参数:
            max_size: 队列最大容量
            feature_dim: 特征向量的维度
            device: 队列存储的设备 (CPU/GPU)
            dtype: 数据类型 (默认float32)
        """
        self.max_size = max_size
        self.feature_dim = feature_dim
        self.device = device
        self.dtype = dtype

        # 预分配固定内存
        self.queue = torch.zeros((max_size, feature_dim), device=device, dtype=dtype)
        self.ptr = 0  # 当前写入位置指针
        self.current_size = 0  # 当前实际大小

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor):
        """
        将一批特征向量加入队列（自动覆盖旧数据）

        参数:
            features: 形状为 [batch_size, feature_dim] 的张量
        """
        batch_size = features.shape[0]

        # 检查输入是否合法
        assert features.shape[1] == self.feature_dim, (
            f"特征维度不匹配! 期望 {self.feature_dim}, 实际 {features.shape[1]}"
        )

        # 计算剩余空间
        remaining = self.max_size - self.ptr

        if batch_size <= remaining:
            # 直接写入连续空间
            self.queue[self.ptr : self.ptr + batch_size] = features
            self.ptr += batch_size
        else:
            # 环形写入：分两部分填充
            self.queue[self.ptr : self.max_size] = features[:remaining]
            self.queue[: batch_size - remaining] = features[remaining:]
            self.ptr = batch_size - remaining

        # 更新当前队列大小（不超过max_size）
        self.current_size = min(self.current_size + batch_size, self.max_size)

    @torch.no_grad()
    def get_queue(self) -> torch.Tensor:
        """获取当前队列的有效内容（按FIFO顺序排列）"""
        if self.current_size < self.max_size:
            return self.queue[: self.current_size]
        else:
            # 如果队列已满，按写入顺序排列（旧数据在前）
            return torch.cat([self.queue[self.ptr :], self.queue[: self.ptr]], dim=0)

    def __len__(self) -> int:
        """返回当前队列中实际存储的样本数"""
        return self.current_size

    def is_full(self) -> bool:
        """队列是否已满"""
        return self.current_size == self.max_size

    def reset(self):
        """清空队列"""
        self.ptr = 0
        self.current_size = 0
