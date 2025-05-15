import lightning
import torch
from .base_handle import BaseHandle
from torch.nn import functional as F
from torchmetrics import Accuracy


class MLMHandle(BaseHandle):
    """
    Handles the model hooks for the MLM task.
    """

    def __init__(self, vocab_size, loss_weight, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

        self.train_accu = Accuracy(
            task="multiclass", num_classes=vocab_size, ignore_index=-100
        )
        self.val_accu = Accuracy(
            task="multiclass", num_classes=vocab_size, ignore_index=-100
        )
        self.loss_weight = loss_weight

    def dispatch_batch(self, batch):
        ids = batch["ids"]
        video = batch["video"]
        video_length = batch["video_length"]
        text = batch["text"]
        text_length = batch["text_length"]

        return ids, video, video_length, text, text_length

    def on_train_epoch_end(self, module):
        """
        Called at the end of the training epoch.
        """
        train_acc = self.train_accu.compute()
        self.log("train_masked_accu", train_acc, prog_bar=True)
        self.train_accu.reset()

    def on_validation_epoch_end(self, module):
        """
        Called at the end of the validation epoch.
        """
        val_acc = self.val_accu.compute()
        self.log("val_masked_accu", val_acc, prog_bar=True)
        self.val_accu.reset()

    @staticmethod
    def generate_padding_attention_mask(
        video_length, text_length, max_video_length=None, max_text_length=None
    ):
        """
        Generate addtivie attention mask for the video and text sequences.
        video_length: [B]
        text_length: [B]
        """
        if max_video_length is None:
            max_video_length = video_length.max().item()
        if max_text_length is None:
            max_text_length = text_length.max().item()

        video_mask = torch.arange(max_video_length, device=video_length.device).expand(
            video_length.size(0), max_video_length
        ) < video_length.unsqueeze(1)
        text_mask = torch.arange(max_text_length, device=text_length.device).expand(
            text_length.size(0), max_text_length
        ) < text_length.unsqueeze(1)

        video_mask = torch.where(video_mask, False, float("-inf"))
        text_mask = torch.where(text_mask, False, float("-inf"))
        return (
            torch.cat((video_mask, text_mask), dim=1).unsqueeze(1).unsqueeze(2)
        )  # for heads dimension and query dimension # (B, 1, 1, L)

    @staticmethod
    def fast_mask_tokens_batch(
        input_ids: torch.Tensor,
        mask_token_id: int,
        mask_ratio=0.15,
        special_token_ids=None,
        pad_token_id=None,
    ):
        """
        高效、向量化地对 (B, L) 的 token ids 批量添加 mask。

        参数：
        - input_ids: (B, L) LongTensor
        - mask_token_id: int，例如 tokenizer.mask_token_id
        - mask_ratio: float
        - special_token_ids: Set[int]
        - pad_token_id: int

        返回：
        - masked_input: (B, L) LongTensor
        - mask_labels: (B, L) LongTensor，其中非 mask 位置为 -100，可直接用于 loss
        """
        B, L = input_ids.shape
        device = input_ids.device

        # 1. 构造 maskable 区域（非特殊 token 且非 padding）
        special_token_ids = set(special_token_ids or [])
        if pad_token_id is not None:
            special_token_ids.add(pad_token_id)

        special_ids_tensor = torch.tensor(list(special_token_ids), device=device)
        is_special = torch.isin(input_ids, special_ids_tensor)  # (B, L)
        maskable = ~is_special  # (B, L)

        # 2. 随机生成 mask 布尔矩阵
        probs = torch.rand(B, L, device=device)
        sampled_mask = (probs < mask_ratio) & maskable  # (B, L)

        # 3. 应用 mask
        masked_input = input_ids.clone()
        masked_input[sampled_mask] = mask_token_id

        # 4. 创建 mask label（非 mask 区域设为 -100，适配 CrossEntropyLoss 忽略）
        mask_labels = torch.full_like(input_ids, fill_value=-100)
        mask_labels[sampled_mask] = input_ids[sampled_mask]

        return masked_input, mask_labels

    def _forward(self, module, video, video_length, text, text_length):
        text_ids = module.tokenizer(text, return_tensors="pt", padding=True)
        text_ids = text_ids["input_ids"].to(module.device)  # (B, L)

        masked_text_ids, mask_text_labels = self.fast_mask_tokens_batch(
            text_ids,
            module.tokenizer.mask_token_id,
            mask_ratio=0.15,
            special_token_ids=module.tokenizer.all_special_ids,
            pad_token_id=module.tokenizer.pad_token_id,
        )

        with torch.no_grad():
            visual_encoder_outputs = module.visual_encoder(video, video_length)
        hidden_state = visual_encoder_outputs.hidden_states
        v_length = visual_encoder_outputs.video_length
        visual_embeddings = module.visual_adapter(hidden_state)  # b t c

        with torch.no_grad():
            textaul_embeddings = module.shared_encoder.get_input_embeddings()(
                masked_text_ids
            )  # b l c

        B, T, C = visual_embeddings.shape
        _, L, _ = textaul_embeddings.shape

        assert T == v_length.max().item(), (
            f"Visual length {T} does not match max video length {v_length.max().item()}"
        )
        assert L == text_length.max().item(), (
            f"Text length {L} does not match max text length {text_length.max().item()}"
        )

        # 5. 生成注意力掩码
        padding_attention_mask = self.generate_padding_attention_mask(
            v_length, text_length
        )
        features = torch.cat((visual_embeddings, textaul_embeddings), dim=1)  # b t+l c

        out_features = module.shared_encoder(
            inputs_embeds=features,
            attention_mask=padding_attention_mask,
        )
        out_features = out_features[:, T:, :]
        out_logit = module.header(out_features)  # b l vocab_size
        return out_logit, mask_text_labels

    def train_step(self, module, batch, batch_idx):
        ids, video, video_length, text, text_length = self.dispatch_batch(batch)

        out_logit, mask_text_labels = self._forward(
            module, video, video_length, text, text_length
        )

        out_loglogit = F.log_softmax(out_logit, dim=-1)

        self.train_accu.update(
            out_loglogit.view(-1, out_loglogit.size(-1)),
            mask_text_labels.view(-1),
            ignore_index=-100,
        )

        loss = (
            F.nll_loss(
                out_loglogit.view(-1, out_loglogit.size(-1)),
                mask_text_labels.view(-1),
                ignore_index=-100,
            )
            * self.loss_weight
        )

        module.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, module, batch, batch_idx):
        ids, video, video_length, text, text_length = self.dispatch_batch(batch)

        out_logit, mask_text_labels = self._forward(
            module, video, video_length, text, text_length
        )

        out_loglogit = F.log_softmax(out_logit, dim=-1)
        self.val_accu.update(
            out_loglogit.view(-1, out_loglogit.size(-1)),
            mask_text_labels.view(-1),
            ignore_index=-100,
        )


if __name__ == "__main__":
    mask = MLMHandle.generate_attention_mask(torch.tensor([5, 3]), torch.tensor([4, 2]))
    print(mask)
