import argparse
import os
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
from PIL import Image
from mmcv import Config
import mmcv

from c3vg.models import build_model
from c3vg.utils import load_checkpoint
from c3vg.core import imshow_box_mask


# Globals initialized after load_model
_GLOBAL_CFG: Config = None
_GLOBAL_TOKENIZER = None


def _ensure_tokenizer(tokenizer_path: str):
    """Load BEiT-3 tokenizer from local spm file."""
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is not None:
        return _GLOBAL_TOKENIZER
    try:
        from transformers import XLMRobertaTokenizer
    except Exception as exc:
        raise RuntimeError("请先安装 transformers 以使用 BEiT-3 tokenizer: pip install transformers") from exc

    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(
            f"未找到 tokenizer 文件: {tokenizer_path}。请参考 README 将 beit3.spm 放在该路径。"
        )
    _GLOBAL_TOKENIZER = XLMRobertaTokenizer(tokenizer_path)
    return _GLOBAL_TOKENIZER


def load_model(config_path: str, checkpoint_path: str, device: str = "cuda:0") -> Any:
    """Load pretrained model according to config and checkpoint."""
    global _GLOBAL_CFG
    cfg = Config.fromfile(config_path)

    # build & load
    model = build_model(cfg.model)
    load_checkpoint(model, load_from=checkpoint_path)

    # move to device + eval
    model.to(device)
    model.eval()

    _GLOBAL_CFG = cfg

    # prepare tokenizer
    # try cfg.model.vis_enc.pretrain tokenizer path (repo default)
    tokenizer_path = os.path.join(os.path.dirname(config_path), "..", "pretrain_weights", "beit3.spm")
    # prefer project-root/pretrain_weights/beit3.spm
    proj_root_tok = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain_weights", "beit3.spm")
    if os.path.isfile(proj_root_tok):
        tokenizer_path = proj_root_tok
    _ensure_tokenizer(os.path.abspath(tokenizer_path))

    return model


def _tokenize(description: str, max_token: int) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenize description using BEiT-3 tokenizer to ids and padding mask.

    Returns:
        ref_expr_inds: shape (max_token,), int
        text_attention_mask: shape (max_token,), int; 0 for tokens, 1 for padding
    """
    tokenizer = _ensure_tokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain_weights", "beit3.spm"))
    tokens = tokenizer.tokenize(description)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    if len(token_ids) == 0:
        raise RuntimeError("文本描述至少需要包含一个 token")

    bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    # Reserve for BOS/EOS
    if len(token_ids) > max_token - 2:
        token_ids = token_ids[: max_token - 2]
    token_ids = [bos] + token_ids + [eos]
    num_tokens = len(token_ids)
    padding_len = max_token - num_tokens
    padding_mask = [0] * num_tokens + [1] * padding_len
    token_ids = token_ids + [pad] * padding_len

    return np.array(token_ids, dtype=np.int64), np.array(padding_mask, dtype=np.int64)


def _preprocess(image: Image.Image, description: str, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """Preprocess PIL image and text to model inputs and img_metas."""
    assert image.mode in ("RGB", "RGBA", "L"), "仅支持 RGB/RGBA/L 模式图片"
    if image.mode == "RGBA":
        image = image.convert("RGB")
    if image.mode == "L":
        image = image.convert("RGB")

    img_np = np.array(image)  # RGB, HxWx3

    # Config knobs
    img_size = int(getattr(cfg, "img_size", 320))
    img_norm_cfg = getattr(cfg, "img_norm_cfg", dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))

    # record ori shape
    h_ori, w_ori = img_np.shape[:2]

    # convert to BGR to mimic pipeline -> then Normalize(to_rgb=True) will swap to RGB
    img_bgr = img_np[:, :, ::-1].copy()

    # Resize to fixed size (keep_ratio=False) like val/test pipeline
    img_resized = mmcv.imresize(img_bgr, (img_size, img_size), return_scale=False)
    h_new, w_new = img_resized.shape[:2]
    scale_factor = np.array([w_new / w_ori, h_new / h_ori, w_new / w_ori, h_new / h_ori], dtype=np.float32)

    # Normalize (to_rgb=True as in config)
    mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
    std = np.array(img_norm_cfg["std"], dtype=np.float32)
    to_rgb = img_norm_cfg.get("to_rgb", True)
    img_norm = mmcv.imnormalize(img_resized, mean, std, to_rgb)

    # Pad to size_divisor=32 (may be no-op)
    img_padded = mmcv.impad_to_multiple(img_norm, 32)

    # HWC->CHW tensor
    img_chw = np.ascontiguousarray(img_padded.transpose(2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).float().unsqueeze(0)  # [1,C,H,W]

    # text
    max_token = int(getattr(cfg, "max_token", 20))
    ref_expr_inds_np, text_attention_mask_np = _tokenize(description, max_token)
    ref_expr_inds = torch.from_numpy(ref_expr_inds_np).long().unsqueeze(0)  # [1, T]
    text_attention_mask = torch.from_numpy(text_attention_mask_np).long().unsqueeze(0)  # [1, T]

    # img_metas
    img_metas = [
        {
            "filename": "",  # 若需要可在 CLI 中传入实际路径
            "expression": description,
            "ori_shape": (h_ori, w_ori, 3),
            "img_shape": img_resized.shape,
            "pad_shape": img_padded.shape,
            "scale_factor": scale_factor,
        }
    ]

    return img_tensor, ref_expr_inds, text_attention_mask, img_metas


def forward(model: Any, image: Image.Image, description: str) -> Dict:
    """对单张图片进行推理并完成后处理，返回可直接用于可视化的结果。

    返回的 Dict 至少包含：
      - pred_bboxes: List[Tensor[x1,y1,x2,y2]]
      - pred_masks: List[RLE dict]
      - pred_bboxes_first / pred_masks_first (如模型提供)
    """
    assert _GLOBAL_CFG is not None, "请先通过 load_model 加载模型以初始化配置"

    img, ref_expr_inds, text_attention_mask, img_metas = _preprocess(image, description, _GLOBAL_CFG)

    model_device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(
            img=img.to(model_device),
            ref_expr_inds=ref_expr_inds.to(model_device),
            img_metas=img_metas,
            text_attention_mask=text_attention_mask.to(model_device),
            return_loss=False,
            rescale=True,
            with_bbox=True,
            with_mask=True,
        )

    # outputs 已由模型内部的 get_predictions 完成后处理（到原图尺寸、阈值化并编码为 RLE）
    return outputs


def _visualize(input_image_path: str, predictions: Dict, output_image_path: str):
    # 取第一张图片的结果进行可视化
    pred_bboxes = predictions.get("pred_bboxes", [None])
    pred_masks = predictions.get("pred_masks", [None])
    pred_bbox = pred_bboxes[0] if len(pred_bboxes) > 0 else None
    pred_mask = pred_masks[0] if len(pred_masks) > 0 else None
    if pred_mask is None:
        raise RuntimeError("未得到分割结果，无法可视化。请确认模型与 checkpoint 是否匹配。")
    imshow_box_mask(input_image_path, pred_bbox, pred_mask, output_image_path, gt=False)


def parse_args():
    parser = argparse.ArgumentParser(description="C3VG 单图推理 Demo")
    parser.add_argument("--input_image_path", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--output_image_path", required=True, type=str)
    # 可选：允许用户指定 config/ckpt
    parser.add_argument("--config", default=os.path.join("configs", "C3VG-Mix.py"), type=str)
    parser.add_argument("--checkpoint", default="", type=str, help="模型 checkpoint 路径")
    parser.add_argument("--device", default="cuda:0", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    # 解析 checkpoint 默认路径（如果未提供）
    ckpt_path = args.checkpoint
    if not ckpt_path:
        # 尝试在项目常见位置查找 segm_best.pth
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "work_dir", "segm_best.pth"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "segm_best.pth"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                ckpt_path = c
                break
        if not ckpt_path:
            raise FileNotFoundError("请通过 --checkpoint 指定模型权重 segm_best.pth 的路径")

    model = load_model(args.config, ckpt_path, device=args.device)

    image = Image.open(args.input_image_path)
    preds = forward(model, image, args.prompt)

    # 保存可视化
    os.makedirs(os.path.dirname(args.output_image_path) or ".", exist_ok=True)
    _visualize(args.input_image_path, preds, args.output_image_path)


if __name__ == "__main__":
    main()


