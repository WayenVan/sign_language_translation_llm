from huggingface_hub import hf_hub_download

# sapiens
hf_hub_download(
    repo_id="noahcao/sapiens-pose-coco",
    subfolder="sapiens_host/pose/checkpoints/sapiens_0.3b",
    filename="sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth",
    local_dir="./outputs",
)
