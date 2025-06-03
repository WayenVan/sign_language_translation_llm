import torch


state_dict = torch.load(
    "outputs/train_pl/2025-06-03_02-47-48/epoch=02-val_llm_generate_bleu=0.07.ckpt"
)["state_dict"]

for key in state_dict.keys():
    if key.startswith("connector."):
        param = state_dict[key]
        print(f"Key: {key}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
        print(f"{param.max()}, {param.min()}, {param.mean()}, {param.std()}")
