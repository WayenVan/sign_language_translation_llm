import torch
from typing import Dict, List
from lightning.pytorch.callbacks import Callback
from torch import Tensor
import lightning.pytorch as pl
import logging
import pprint
from collections import defaultdict


def check_nan_in_model_params(model: torch.nn.Module) -> Dict[str, List[str]]:
    """
    检查模型中是否存在NaN值的参数或梯度
    返回字典结构: {
        'nan_params': [存在NaN的参数名列表],
        'nan_grads': [存在NaN梯度的参数名列表]
    }
    """
    result = defaultdict(list)
    for name, param in model.named_parameters():
        # 检查参数本身
        if torch.isnan(param).any():
            result["nan_params"].append(name)

        # 检查梯度（需确保已计算梯度）
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                result["nan_grads"].append(name)

    return result


class NanDetectCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.hooks = []
        self._loggger = logging.getLogger("nan_detect_callback")

    def on_before_backward(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: Tensor
    ) -> None:
        checked_nan = check_nan_in_model_params(pl_module)
        if checked_nan["nan_params"] or checked_nan["nan_grads"]:
            self._loggger.warning(
                f"NaN detected in model parameters or gradients: {pprint.pformat(checked_nan)}"
            )
            self._loggger.warning(
                "Stopping training due to NaN values in model parameters or gradients."
            )
            trainer.should_stop = True

        return super().on_before_backward(trainer, pl_module, loss)
