import torch

ckpt_path = "work_dir/unimodel/pretrain/AAAI/uni-320/20241210_130152/segm_best.pth"
ckpt = torch.load(ckpt_path)
ckpt.pop("optimizer")
ckpt.pop("scheduler")

torch.save(ckpt, "work_dir/unimodel/pretrain/AAAI/model_results/model.pth")