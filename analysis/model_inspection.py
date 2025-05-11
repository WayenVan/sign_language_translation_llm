import sys
import os
import logging
import re

from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # NOTE: this is the initial cwd when runing the sciprt, the hydra will change the cwd to the output dir

import hydra
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


from lightning import Trainer
from lightning.pytorch import callbacks
import lightning.pytorch as pl

from model.slt import SLTModel
import cv2
from dataclasses import dataclass, field


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
logger = logging.getLogger(__name__)  # NOTE: lightning already setupo the logger for us
cv2.setNumThreads(0)  # NOTE: set the number of threads to 0 to avoid cv2 error


@dataclass
class Config:
    ckpt: str = "/root/projects/slt_set_llms/outputs/train/2025-05-10_13-24-20/epoch=epoch=08-wer=val_token_level_accu=0.49.ckpt"
    model_cfg: str = "outputs/train/2025-05-10_13-24-20/.hydra/config.yaml"
    selected_samples: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    hydra: DictConfig = field(
        default_factory=lambda: OmegaConf.create(
            {"run": {"dir": "outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}"}}
        )
    )


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


# NOTE: the hydra appp only inisitalize once
@hydra.main(config_name="config", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    train(cfg)


def train(cfg: DictConfig) -> None:
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    working_dir = hydra_config.runtime.output_dir
    config_name = hydra_config.job.config_name

    model_cfg = OmegaConf.load(cfg.model_cfg)
    selected_samples = cfg.selected_samples

    logger.info(f"Output directory: {working_dir}")

    # NOTE: load vocab
    with open(model_cfg.data.vocab_file, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]  # Remove empty lines

    # NOTE: define callbacks for trainer
    cbs = [
        callbacks.RichProgressBar(),
        InspectionCallback(),
    ]

    # NOTE: start training
    t = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=[0],
        callbacks=cbs,
        log_every_n_steps=50,
        max_steps=4,
        max_epochs=model_cfg.max_epochs,
        gradient_clip_val=0.5,  # NOTE: gradient clipping will be normed
        gradient_clip_algorithm="value",
        sync_batchnorm=True,
        precision="16-mixed",
        logger=None,
        # detect_anomaly=True,
    )

    logger.info(f"Process in local rank {t.local_rank}, global rank {t.global_rank}")

    # NOTE: create a subset only containing selected samples
    datamodule = instantiate(model_cfg.data.datamodule, model_cfg)
    datamodule.setup()

    val_set = datamodule.val_dataset
    val_subset = Subset(val_set, selected_samples)
    loader_params = model_cfg.data.datamodule
    loader_params = OmegaConf.to_container(loader_params, resolve=True)
    del loader_params["_target_"]
    loader_params["num_workers"] = 1
    loader_params["batch_size"] = 1

    dataloader = DataLoader(
        val_subset, **loader_params, collate_fn=datamodule.collate_fn
    )

    model = SLTModel.load_from_checkpoint(cfg.ckpt, vocab=vocab, cfg=model_cfg)
    t.validate(model, dataloaders=dataloader)
    attns = cbs[1].cross_attention_collection
    logger.info(f"Cross attention weights: {attns}")
    for i, attn in enumerate(attns):
        attn = attn.squeeze(0).cpu().numpy()
        plt.imshow(attn, cmap="viridis", interpolation="nearest")
        plt.colorbar()
        plt.savefig(f"{working_dir}/attn_{i}.png")


class InspectionCallback(callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()

        self.cross_attention_collection = []
        self.get_cross_attention_hooks = []
        self.change_decoder_param_hooks = []

    def get_cross_attention_hook(self, module, input, output):
        """
        the return of the llamma decoder:
        return self.LlamaCrossDecoderOutputs(
            logits=logits,
            token_llm_features=llm_embeds,
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attns,
        )
        """
        cross_attention = output.cross_attentions[
            -1
        ]  # the last cross attention weights
        self.cross_attention_collection.append(cross_attention)

    def setup(self, trainer, pl_module: "pl.LightningModule", stage: str) -> None:
        # NOTE: register hooks for the model
        for name, module in pl_module.named_modules():
            if name == "decoder":
                # NOTE: change the decoder to output cross attention weights
                original_forward = module.forward

                def wrapped_call(*args, **kwargs):
                    # NOTE: change the decoder to output cross attention weights
                    kwargs["output_attentions"] = True
                    return original_forward(*args, **kwargs)

                module.forward = wrapped_call
                module.register_forward_hook(self.get_cross_attention_hook)


if __name__ == "__main__":
    main()
