import math
import warnings
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from mmrotate.registry import MODELS
from .utils import weighted_loss

@weighted_loss
def CircumSIoULoss(pred, target, eps=1e-7, neg_gamma=False):

    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    # modified clamp threshold zero to eps to avoid NaN
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=eps)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # angle cost
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps

    sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)

    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = torch.cos(torch.asin(sin_alpha) * 2 - math.pi / 2)

    # distance cost
    rho_x = (s_cw / cw)**2
    rho_y = (s_ch / ch)**2

    # `neg_gamma=True` follows original implementation in paper
    # but setting `neg_gamma=False` makes training more stable.
    gamma = angle_cost - 2 if neg_gamma else 2 - angle_cost
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

    # shape cost
    omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(
        1 - torch.exp(-1 * omiga_h), 4)

    # SIoU
    sious = ious - 0.5 * (distance_cost + shape_cost)
    loss = 1 - sious.clamp(min=-1.0, max=1.0)
    return loss

@MODELS.register_module()
class SIoULoss(nn.Module):
    r"""`Implementation of paper `SIoU Loss: More Powerful Learning
    for Bounding Box Regression <https://arxiv.org/abs/2205.12740>`_.

    Code is modified from https://github.com/meituan/YOLOv6.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
        neg_gamma (bool): `True` follows original implementation in paper.

    Return:
        Tensor: Loss tensor.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 neg_gamma: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.neg_gamma = neg_gamma

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * CircumSIoULoss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            neg_gamma=self.neg_gamma,
            **kwargs)
        return loss
