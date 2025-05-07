import cv2
import mmcv
import numpy
import torch
import os.path as osp

from c3vg.core.utils import boxsegiou, imshow_box_mask
from c3vg.models.heads.uni_head import compute_boxiou
from c3vg.utils import load_checkpoint, get_root_logger
from c3vg.core import imshow_expr_bbox, imshow_expr_mask
from c3vg.models import build_model, ExponentialMovingAverage
from c3vg.datasets import extract_data, build_dataset, build_dataloader
# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

try:
    import apex
except:
    pass


def inference_model(cfg):
    datasets_cfg = [cfg.data.train]
    for which_set in cfg.which_set:
        datasets_cfg.append(eval(f"cfg.data.{which_set}"))

    datasets = list(map(build_dataset, datasets_cfg))
    dataloaders = list(map(lambda dataset: build_dataloader(cfg, dataset), datasets))

    model = build_model(cfg.model, word_emb=datasets[0].word_emb, num_token=datasets[0].num_token)
    model = model.cuda()
    if cfg.use_fp16:
        model = apex.amp.initialize(model, opt_level="O1")
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True
    if cfg.ema:
        model_ema = ExponentialMovingAverage(model, cfg.ema_factor)
    else:
        model_ema = None
    load_checkpoint(model, model_ema, None, cfg.checkpoint)
    if cfg.ema:
        model_ema.apply_shadow()

    model.eval()
    logger = get_root_logger()
    with_bbox, with_mask = False, False
    for i, which_set in enumerate(cfg.which_set):
        logger.info(f"inferencing on split {which_set}")
        prog_bar = mmcv.ProgressBar(len(datasets[i + 1]))
        with torch.no_grad():
            for batch, inputs in enumerate(dataloaders[i + 1]):
                gt_bbox, gt_mask, is_crowd = None, None, None
                if "gt_bbox" in inputs:
                    with_bbox = True
                    gt_bbox = inputs.pop("gt_bbox").data[0]
                if "gt_mask_rle" in inputs:
                    with_mask = True
                    gt_mask = inputs.pop("gt_mask_rle").data[0]
                if "is_crowd" in inputs:
                    inputs.pop("is_crowd").data[0]

                if not cfg.distributed:
                    inputs = extract_data(inputs)

                img_metas = inputs["img_metas"]
                batch_size = len(img_metas)

                predictions = model(**inputs, return_loss=False, rescale=True, with_bbox=with_bbox, with_mask=with_mask)

                pred_bboxes = [None for _ in range(batch_size)]
                if with_bbox:
                    pred_bboxes = predictions.pop("pred_bboxes")
                    pred_bboxes_first = predictions.pop("pred_bboxes_first")
                if cfg["dataset"] == "GRefCOCO":
                    tmp_pred_bboxes = []
                    for pred_bbox in pred_bboxes:
                        img_level_bboxes = pred_bbox["boxes"]
                        scores = pred_bbox["scores"]
                        keep_ind = scores > cfg.score_threahold
                        img_level_bboxes = img_level_bboxes[keep_ind]
                        tmp_pred_bboxes.append(img_level_bboxes)
                    pred_bboxes = tmp_pred_bboxes

                pred_masks = [None for _ in range(batch_size)]
                if with_mask:
                    pred_masks = predictions.pop("pred_masks")
                    pred_masks_first = predictions.pop("pred_masks_first")
                    
                for j, (img_meta, pred_bbox, pred_mask, pred_bbox_first, pred_mask_first) in enumerate(zip(img_metas, pred_bboxes, pred_masks, pred_bboxes_first, pred_masks_first)):
                    filename, expression = img_meta["filename"], img_meta["expression"]
                    bbox_gt, mask_gt = None, None
                    if cfg.with_gt and with_bbox:
                        bbox_gt = gt_bbox[j]
                    if cfg.with_gt and with_mask:
                        mask_gt = img_meta["gt_ori_mask"]

                    scale_factors = img_meta["scale_factor"]
                    # pred_bbox /= pred_bbox.new_tensor(scale_factors)
                    
                    outfile = osp.join(cfg.output_dir, cfg.dataset + "_" + which_set, expression.replace(" ", "_") + "_" + osp.basename(filename).split(".jpg")[0])
                    badcase=False
                    bbox_gt /= bbox_gt.new_tensor(scale_factors)
                    # if boxsegiou(bbox_gt, pred_mask_first)<0.5 and boxsegiou(bbox_gt, pred_mask)>0.7:
                    # if boxsegiou(pred_bbox, mask_gt)-boxsegiou(pred_bbox_first, mask_gt)>0.05 and compute_boxiou(bbox_gt, pred_bbox)>0.9:
                    if compute_boxiou(bbox_gt.unsqueeze(0), pred_bbox.unsqueeze(0).cpu()).squeeze()>0.9 and compute_boxiou(bbox_gt.unsqueeze(0), pred_bbox_first.unsqueeze(0).cpu()).squeeze()- compute_boxiou(bbox_gt.unsqueeze(0), pred_bbox.unsqueeze(0).cpu()).squeeze() < -0.1:
                    # if compute_boxiou(bbox_gt.unsqueeze(0), pred_bbox.unsqueeze(0).cpu()).squeeze()<0.3:
                        badcase = True
                    
                    if not cfg.onlybadcase or (cfg.onlybadcase and badcase):
                        # box seg分开绘制
                        # if with_bbox:
                        #     bbox_gt /= bbox_gt.new_tensor(scale_factors)
                        #     outfile_det = outfile + "_box_fine.jpg"
                        #     imshow_expr_bbox(filename, pred_bbox, outfile_det, gt_bbox=None)
                        #     outfile_det = outfile + "_box_course.jpg"
                        #     imshow_expr_bbox(filename, pred_bbox_first, outfile_det, gt_bbox=None)
                        # if with_mask:
                        #     import pycocotools.mask as maskUtils
                        #     outfile_seg = outfile + "_seg_fine.jpg"
                        #     # imshow_expr_mask(filename, pred_mask, outfile_seg, gt_mask=mask_gt, overlay=cfg.overlay)
                        #     imshow_expr_mask(filename, pred_mask, outfile_seg, gt_mask=None, overlay=cfg.overlay)
                        #     outfile_seg = outfile + "_seg_course.jpg"
                        #     imshow_expr_mask(filename, pred_mask_first, outfile_seg, gt_mask=None, overlay=cfg.overlay)
                        #     outfile_seg = outfile + "_only_mask_course.jpg"
                        #     cv2.imwrite(outfile_seg, (maskUtils.decode(pred_mask_first)*255).astype(numpy.uint8))
                        #     outfile_seg = outfile + "_only_mask_fine.jpg"
                        #     cv2.imwrite(outfile_seg, (maskUtils.decode(pred_mask)*255).astype(numpy.uint8))
                        
                        # boxseg合并绘制
                        outfile_pred = outfile+"_pred.jpg"
                        imshow_box_mask(filename, pred_bbox, pred_mask, outfile_pred, gt=False)
                        
                        outfile_pred = outfile+"_pred_course.jpg"
                        imshow_box_mask(filename, pred_bbox_first, pred_mask_first, outfile_pred, gt=False)
                        
                        # bbox_gt /= bbox_gt.new_tensor(scale_factors)
                        outfile_gt = outfile+"_gt.jpg"
                        imshow_box_mask(filename, bbox_gt, mask_gt, outfile_gt, gt=True)

                    prog_bar.update()
    if cfg.ema:
        model_ema.restore()
