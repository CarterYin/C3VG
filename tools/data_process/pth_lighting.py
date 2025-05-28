import torch

ckpt_path = "work_dir/seed/C3VG-1234/20250527_002219/segm_best.pth"
ckpt = torch.load(ckpt_path)
ckpt.pop("optimizer")
ckpt.pop("scheduler")

torch.save(ckpt, "work_dir/unimodel/pretrain/AAAI/lighting/model.pth")