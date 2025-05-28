import torch
from torch import nn
from torch.nn import functional as F
import torch
from c3vg.models import HEADS
import pycocotools.mask as maskUtils
import numpy as np

from .unet_head import *
from c3vg.models.losses.boxloss import BoxLoss
from .modules import BoxSegAttention, BoxSegPooler, QueryAugment, SegBranch, BoxBranch, UnifiedInteractionModule
from .unet_head import SimpleFPN
from ..utils import xywh_to_x1y1x2y2, x1y1x2y2_to_xywh
from ..losses.clip_loss import ClipLoss, get_rank, get_world_size
from .projection import Projector


@HEADS.register_module()
class UniHeadCoarseToFine(nn.Module):
    def __init__(
        self,
        input_channels=768,
        hidden_channels=256,
        query_augment=None,
        loss_weight={"mask": 1.0, "bbox": 0.025, "cons": 0.0},
        threshold={"B2S": 0.1},
        start_epoch=0,
        decoder_upsample_type="none",
        uim={"enable": False, "weighted_compose": "none", "enable_box_coorinate_embed": False, "box_weights": [0.1, 1.0]},
    ):
        super(UniHeadCoarseToFine, self).__init__()
        self.seg_branch_first = SegBranch(hidden_channels, upsample_rate=1)
        self.box_branch_first = BoxBranch(hidden_channels)
        self.decoder_upsample_type = decoder_upsample_type
        if self.decoder_upsample_type == "fpn":
            self.neck = SimpleFPN(
                backbone_channel=hidden_channels,
                in_channels=[hidden_channels // 4, hidden_channels // 2, hidden_channels, hidden_channels],
                out_channels=[hidden_channels // 4, hidden_channels // 2, hidden_channels, hidden_channels * 2],
            )
            self.fpn_decoder = SimpleDecoding(hidden_channels * 2)
        elif self.decoder_upsample_type == "tranposeconv":
            self.neck = SegBranch(hidden_channels, upsample_rate=4).neck
        else:
            self.neck = nn.Identity()
        self.seg_branch_second = SegBranch(hidden_channels, upsample_rate=1)
        self.box_branch_second = BoxBranch(hidden_channels)
        self.box_loss = BoxLoss()
        self.seg_loss = nn.functional.cross_entropy
        self.loss_weight = loss_weight
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.boxsegattn = BoxSegAttention(input_channels=hidden_channels, input_size=input_size)
        self.clip_loss = ClipLoss(rank=get_rank(), world_size=get_world_size())
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.lan_embedding = nn.Linear(input_channels, hidden_channels, bias=False)
        self.img_embedding = nn.Conv2d(input_channels, hidden_channels, kernel_size=1, bias=False)
        self.query_embedding = nn.Linear(input_channels, hidden_channels, bias=False)

        self.seg_cons_embedding = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.box_cons_embedding = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.boxsegpooler = BoxSegPooler()
        self.proj_pixel_level_cons = Projector(word_dim=hidden_channels, in_dim=hidden_channels, hidden_dim=hidden_channels // 2, kernel_size=1)
        # query augment module
        self.query_augment_module = None
        if query_augment is not None:
            self.query_augment_module = QueryAugment(hidden_channels=hidden_channels, num_queries=query_augment["num_queries"])
        self.threshold = threshold
        self.start_epoch = start_epoch
        self.unified_interaction_module = uim["enable"]
        if self.unified_interaction_module:
            self.UIM = UnifiedInteractionModule(
                input_channels=hidden_channels,
                box_weights=uim["box_weights"],
                weighted_compose=uim["weighted_compose"],
                enable_box_coorinate_embed=uim["enable_box_coorinate_embed"],
            )

    def text_pooler(self, lan_feat, lan_mask):
        lan_feat_pooler = torch.cat(list(map(lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0], lan_feat, ~lan_mask)))
        return lan_feat_pooler

    def forward_train(self, x, targets, cls_feat=None, lan_feat=None, lan_mask=None, img=None):
        device = x.device
        target_mask = torch.from_numpy(np.concatenate([maskUtils.decode(target)[None] for target in targets["mask"]])).to(device)
        # all feats embedding to hidden_channels
        img_feat = self.img_embedding(x)
        query_feat = self.query_embedding(cls_feat)
        lan_feat = self.lan_embedding(lan_feat)
        lan_pool = self.text_pooler(lan_feat, lan_mask)
        # ! query augment
        if self.query_augment_module is not None:
            query_feat = self.query_augment_module(query_feat, img_feat, lan_feat, lan_mask)
        # ! stage 1
        pred_bbox = self.box_branch_first(query_feat)
        if self.loss_weight["clip"]["pixel"]:
            pred_mask = self.proj_pixel_level_cons(img_feat, lan_pool)
        else:
            pred_mask = self.seg_branch_first(img_feat)
        pred_mask_first = F.interpolate(pred_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)
        pred_bbox_first = pred_bbox
        target_mask_first = target_mask

        # # * first stage -- box seg pooling -> box_feat(B,C) seg_feat(B,C)
        # # ! box seg pooling
        # box_feat_first, [seg_feat_pos_first, seg_feat_neg_first] = self.boxsegpooler(
        #     pred_bbox,
        #     pred_mask,
        #     img_feat,
        #     gt=False,
        #     img_pool=False,
        # )

        # ! stage 2
        # ! unified interaction
        extra_dict = {}
        if self.unified_interaction_module:
            # pred_bbox_2_tmp, pred_mask_2_tmp = self.boxsegattn(box_feat_first, seg_feat_pos_first, img_feat, lan_feat, lan_mask)
            pred_bbox_2_tmp, pred_mask_2_tmp, extra = self.UIM(pred_bbox, pred_mask, img_feat, lan_feat, lan_mask)
            if "unified_img_feat" in extra:
                extra_dict["unified_img_feat"] = F.interpolate(extra["unified_img_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
            if "seg_feat" in extra:
                extra_dict["seg_feat"] = F.interpolate(extra["seg_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
            if "box_feat" in extra:
                extra_dict["box_feat"] = F.interpolate(extra["box_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
            if "img_feat" in extra:
                extra_dict["img_feat"] = F.interpolate(extra["img_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
        else:
            pred_bbox_2_tmp, pred_mask_2_tmp = query_feat, img_feat

        # ! decoder upsample
        if self.decoder_upsample_type == "fpn":
            x_c1, x_c2, x_c3, x_c4 = self.neck(pred_mask_2_tmp)
            pred_mask_up4 = self.fpn_decoder(x_c4, x_c3, x_c2, x_c1)
        elif self.decoder_upsample_type == "tranposeconv":
            pred_mask_up4 = self.neck(pred_mask_2_tmp)
        else:
            pred_mask_up4 = pred_mask_2_tmp

        # ! pixel level cons
        if self.loss_weight["clip"]["pixel"]:
            pred_mask_2 = self.proj_pixel_level_cons(pred_mask_up4, lan_pool)
        else:
            pred_mask_2 = self.seg_branch_second(pred_mask_up4)
        pred_bbox_second = self.box_branch_second(pred_bbox_2_tmp)
        pred_mask_second = F.interpolate(pred_mask_2, size=img.shape[-2:], mode="bilinear", align_corners=True)

        # ! loss func
        loss_mask_first = seg_loss(pred_mask_first, target_mask_first, self.loss_weight["mask"]) * self.loss_weight["stage"]["first"]
        loss_bbox_first = bbox_loss(pred_bbox_first, targets["bbox"], img, self.box_loss) * self.loss_weight["bbox"] * self.loss_weight["stage"]["first"]
        loss_mask_second = seg_loss(pred_mask_second, target_mask, self.loss_weight["mask"]) * self.loss_weight["stage"]["second"]
        loss_bbox_second = bbox_loss(pred_bbox_second, targets["bbox"], img, self.box_loss) * self.loss_weight["bbox"] * self.loss_weight["stage"]["second"]

        # ! cons loss
        loss_cons_first = torch.tensor([0.0], device=device)
        loss_cons_second = torch.tensor([0.0], device=device)
        if (self.loss_weight["boxsegcc"]["S2B"] + self.loss_weight["boxsegcc"]["B2S"] > 0) and targets["epoch"] > self.start_epoch:
            loss_S2B_first, loss_B2S_first, _, _ = boxseg_iouloss2(
                pred_bbox_first, pred_mask_first, B2Sthr=self.threshold["B2S"], S2Bthr=self.threshold["S2B"], box_func=self.box_loss
            )
            loss_S2B_second, loss_B2S_second, _, _ = boxseg_iouloss2(
                pred_bbox_second, pred_mask_second, B2Sthr=self.threshold["B2S"], S2Bthr=self.threshold["S2B"], box_func=self.box_loss
            )
            loss_cons_first = (
                sum(loss_S2B_first) / len(loss_S2B_first) * self.loss_weight["boxsegcc"]["S2B"]
                + sum(loss_B2S_first) / len(loss_B2S_first) * self.loss_weight["boxsegcc"]["B2S"]
            ) * self.loss_weight["stage"]["first"] * 0.1
            loss_cons_second = (
                sum(loss_S2B_second) / len(loss_S2B_second) * self.loss_weight["boxsegcc"]["S2B"]
                + sum(loss_B2S_second) / len(loss_B2S_second) * self.loss_weight["boxsegcc"]["B2S"]
            ) * self.loss_weight["stage"]["second"] * 0.1

        loss_mask = loss_mask_first + loss_mask_second
        loss_det = loss_bbox_first + loss_bbox_second
        loss_cons = loss_cons_first + loss_cons_second
        loss_dict = {
            "loss_mask": loss_mask,
            "loss_det": loss_det,
            "loss_cons": loss_cons,
            "loss_mask_first": loss_mask_first,
            "loss_bbox_first": loss_bbox_first,
            "loss_mask_second": loss_mask_second,
            "loss_bbox_second": loss_bbox_second,
            "loss_cons_first": loss_cons_first,
            "loss_cons_second": loss_cons_second,
        }
        pred_dict = {"pred_mask": pred_mask_second, "pred_bbox": pred_bbox_second, "pred_mask_first": pred_mask_first, "pred_bbox_first": pred_bbox_first}
        return loss_dict, pred_dict, extra_dict

    def forward_test(self, x, cls_feat=None, lan_feat=None, lan_mask=None, img=None, targets=None):
        img_feat = self.img_embedding(x)
        query_feat = self.query_embedding(cls_feat)
        lan_feat = self.lan_embedding(lan_feat)
        lan_pool = self.text_pooler(lan_feat, lan_mask)
        # ! query augment
        if self.query_augment_module is not None:
            query_feat = self.query_augment_module(query_feat, img_feat, lan_feat, lan_mask)
        # ! stage 1
        pred_bbox = self.box_branch_first(query_feat)
        if self.loss_weight["clip"]["pixel"]:
            pred_mask = self.proj_pixel_level_cons(img_feat, lan_pool)
        else:
            pred_mask = self.seg_branch_first(img_feat)
        pred_mask_first = F.interpolate(pred_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)
        pred_bbox_first = pred_bbox

        # # * first stage -- box seg pooling -> box_feat(B,C) seg_feat(B,C)
        # # ! box seg pooling
        # box_feat_first, [seg_feat_pos_first, seg_feat_neg_first] = self.boxsegpooler(
        #     pred_bbox,
        #     pred_mask,
        #     img_feat,
        #     gt=False,
        #     img_pool=False,
        # )

        # ! stage 2
        # ! unified interaction
        extra_dict = {}
        if self.unified_interaction_module:
            # pred_bbox_2_tmp, pred_mask_2_tmp = self.boxsegattn(box_feat_first, seg_feat_pos_first, img_feat, lan_feat, lan_mask)
            pred_bbox_2_tmp, pred_mask_2_tmp, extra = self.UIM(pred_bbox, pred_mask, img_feat, lan_feat, lan_mask)
            if "unified_img_feat" in extra:
                extra_dict["unified_img_feat"] = F.interpolate(extra["unified_img_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
            if "seg_feat" in extra:
                extra_dict["seg_feat"] = F.interpolate(extra["seg_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
            if "box_feat" in extra:
                extra_dict["box_feat"] = F.interpolate(extra["box_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
            if "img_feat" in extra:
                extra_dict["img_feat"] = F.interpolate(extra["img_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
        else:
            pred_bbox_2_tmp, pred_mask_2_tmp = query_feat, img_feat

        # ! decoder upsample
        if self.decoder_upsample_type == "fpn":
            x_c1, x_c2, x_c3, x_c4 = self.neck(pred_mask_2_tmp)
            pred_mask_up4 = self.fpn_decoder(x_c4, x_c3, x_c2, x_c1)
        elif self.decoder_upsample_type == "tranposeconv":
            pred_mask_up4 = self.neck(pred_mask_2_tmp)
        else:
            pred_mask_up4 = pred_mask_2_tmp

        # ! pixel level cons
        if self.loss_weight["clip"]["pixel"]:
            pred_mask_2 = self.proj_pixel_level_cons(pred_mask_up4, lan_pool)
        else:
            pred_mask_2 = self.seg_branch_second(pred_mask_up4)
        pred_bbox_second = self.box_branch_second(pred_bbox_2_tmp)
        pred_mask_second = F.interpolate(pred_mask_2, size=img.shape[-2:], mode="bilinear", align_corners=True)

        pred_dict = {"pred_mask": pred_mask_second, "pred_bbox": pred_bbox_second, "pred_mask_first": pred_mask_first, "pred_bbox_first": pred_bbox_first}
        return pred_dict, extra_dict

def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def sigmoid_ce_loss(inputs, targets):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
    return ce_loss


def seg_loss(inputs, target, loss_info):
    # weight = torch.FloatTensor([0.9, 1.1]).cuda()
    # return loss_func(inputs, target.long(), weight=weight)
    loss_seg = torch.tensor([0.0], device=inputs.device)
    target = target.float().unsqueeze(1)
    assert target.shape == inputs.shape
    if "dice" in loss_info:
        loss_seg += dice_loss(inputs, target) * loss_info["dice"]
    if "bce" in loss_info:
        loss_seg += sigmoid_ce_loss(inputs, target) * loss_info["bce"]

    return loss_seg


def box_norm(targets, img):
    gt_bbox = torch.stack(targets, dim=0)
    norm_bbox = torch.zeros_like(gt_bbox, device=gt_bbox.device)

    norm_bbox[:, 0] = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2.0  # x_center
    norm_bbox[:, 1] = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2.0  # y_center
    norm_bbox[:, 2] = gt_bbox[:, 2] - gt_bbox[:, 0]  # w
    norm_bbox[:, 3] = gt_bbox[:, 3] - gt_bbox[:, 1]  # h

    img_size = torch.tensor(img.shape[-2:], device=img.device)
    norm_bbox = norm_bbox / (img_size.unsqueeze(0).repeat((img.shape[0], 2)))
    return norm_bbox


def bbox_loss(inputs, targets, img, loss_func):
    norm_bbox = box_norm(targets, img)
    loss, loss_box, loss_giou = loss_func(inputs, norm_bbox)
    return loss


def consistencyloss(bbox_feat, seg_feat, loss_func):
    # consistencyloss
    label = torch.arange(bbox_feat.shape[0])
    feats = torch.concat((bbox_feat, seg_feat), dim=0)
    labels = torch.concat((label, label))
    consloss = loss_func(feats, labels)
    return consloss


def clip_infonNCEloss(feat1, feat2, loss_func, logit_scale):
    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1)
    return loss_func(feat1, feat2, logit_scale.exp())


def get_maskouterbox(mask_input, threshold=0.5):
    """
    计算二值化mask的外接矩形（bounding box）。

    Args:
        mask (torch.Tensor): 二值化的mask，形状为 (B, H, W)，由0和1组成。

    Returns:
        torch.Tensor: 外接矩形的坐标 (x1, y1, x2, y2)，形状为 (B, 4)。
    """
    B, H, W = mask_input.shape
    mask = (mask_input > threshold).int()
    # mask = mask_input

    # 找到每个batch中mask的非零元素的坐标
    y_coords = torch.arange(H, device=mask.device).view(1, -1, 1).expand(B, H, W) * mask
    x_coords = torch.arange(W, device=mask.device).view(1, 1, -1).expand(B, H, W) * mask

    if mask.sum() == 0:
        return torch.zeros((1, 4), device=mask.device)
    else:
        # 获取外接矩形的边界
        x1 = x_coords.masked_select(mask.bool()).view(B, -1).min(dim=1)[0]
        y1 = y_coords.masked_select(mask.bool()).view(B, -1).min(dim=1)[0]
        x2 = x_coords.masked_select(mask.bool()).view(B, -1).max(dim=1)[0]
        y2 = y_coords.masked_select(mask.bool()).view(B, -1).max(dim=1)[0]

    return torch.stack([x1, y1, x2, y2], dim=1)


def compute_boxiou(box1, box2):
    """
    计算两个矩形框的IoU

    Args:
        box1 (torch.Tensor): 第一个矩形框的坐标 (B, 4)
        box2 (torch.Tensor): 第二个矩形框的坐标 (B, 4)

    Returns:
        torch.Tensor: IoU值，形状为 (B, )
    """
    x1_inter = torch.max(box1[:, 0], box2[:, 0])
    y1_inter = torch.max(box1[:, 1], box2[:, 1])
    x2_inter = torch.min(box1[:, 2], box2[:, 2])
    y2_inter = torch.min(box1[:, 3], box2[:, 3])

    inter_area = torch.clamp(x2_inter - x1_inter + 1, min=0) * torch.clamp(y2_inter - y1_inter + 1, min=0)

    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area.clamp(min=1e-6)
    return iou


def compute_boxloss(maskouter, predbox, loss_func):
    maskouter = x1y1x2y2_to_xywh(maskouter)
    predbox = x1y1x2y2_to_xywh(predbox)
    loss, loss_box, loss_giou = loss_func(maskouter, predbox)
    return loss


def compute_segboxiou(seg, box, threshold=0.5):
    seg = (seg > threshold).int()
    # Extract the sub-mask corresponding to the box
    x1, y1, x2, y2 = box
    sub_mask = seg[y1:y2, x1:x2]
    # Calculate the intersection
    intersection = sub_mask.sum().float()
    # Calculate the area of the mask
    mask_area = seg.sum().float()
    # box_area = abs(x2 - x1) * abs(y2 - y1)
    # Calculate the IoU
    if mask_area > 0:
        iou = intersection / mask_area
    else:
        iou = torch.tensor(0.0, device=seg.device)
    return iou


def boxseg_iouloss(box, mask, B2Sthr=0.5, S2Bthr=0.5, box_func=None):
    # xywh_to_x1y1x2y2
    box_x1y1x2y2 = xywh_to_x1y1x2y2(box)
    mask = mask.sigmoid().squeeze(1)
    B, H, W = mask.shape
    loss_S2B, loss_B2S = [], []
    S2B_iou_list, B2S_iou_list = [], []
    for b in range(B):
        x1, y1, x2, y2 = (box_x1y1x2y2[b] * torch.tensor([W, H, W, H], device=box.device)).to(torch.int)
        x1, x2 = torch.clamp(x1, 0, W), torch.clamp(x2, 0, W)
        y1, y2 = torch.clamp(y1, 0, H), torch.clamp(y2, 0, H)
        box_pred = torch.tensor([x1, y1, x2, y2], device=box.device)
        S2B_iou = compute_segboxiou(mask[b], box_pred, threshold=S2Bthr)
        maskouterbbox = get_maskouterbox(mask[b].unsqueeze(0), threshold=B2Sthr).squeeze(0)
        B2S_iou = compute_boxiou(maskouterbbox.unsqueeze(0), box_pred.unsqueeze(0)).squeeze(0)
        loss_S2B.append(1 - S2B_iou)
        loss_B2S.append(1 - B2S_iou)
        S2B_iou_list.append(S2B_iou)
        B2S_iou_list.append(B2S_iou)

    return loss_S2B, loss_B2S, S2B_iou_list, B2S_iou_list


def boxseg_iouloss2(box, mask, B2Sthr=0.5, S2Bthr=0.5, box_func=BoxLoss()):
    # xywh_to_x1y1x2y2
    box_x1y1x2y2 = xywh_to_x1y1x2y2(box)
    mask = mask.sigmoid().squeeze(1)
    B, H, W = mask.shape
    loss_S2B, loss_B2S = [], []
    for b in range(B):
        x1, y1, x2, y2 = (box_x1y1x2y2[b] * torch.tensor([W, H, W, H], device=box.device)).to(torch.int)
        x1, x2 = torch.clamp(x1, 0, W), torch.clamp(x2, 0, W)
        y1, y2 = torch.clamp(y1, 0, H), torch.clamp(y2, 0, H)
        box_pred = torch.tensor([x1, y1, x2, y2], device=box.device)
        S2B_iou = compute_segboxiou(mask[b], box_pred, threshold=S2Bthr)
        maskouterbbox = get_maskouterbox(mask[b].unsqueeze(0), threshold=B2Sthr).squeeze(0)
        maskouterbbox_norm = maskouterbbox / torch.tensor([W, H, W, H], device=box.device)
        B2S_loss = compute_boxloss(maskouterbbox_norm.unsqueeze(0), box_x1y1x2y2[b].unsqueeze(0), loss_func=box_func)
        loss_S2B.append(1 - S2B_iou)
        loss_B2S.append(B2S_loss)

    return loss_S2B, loss_B2S, None, None
