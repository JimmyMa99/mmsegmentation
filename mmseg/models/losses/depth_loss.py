# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss

import pdb
from .utils import weighted_loss

def eps_lossfn(cam, saliency, num_classes, label, tau, lam, intermediate=True):
    """
    Get EPS loss for pseudo-pixel supervision from saliency map.
    Args:
        cam (tensor): response from model with float values.
        saliency (tensor): saliency map from off-the-shelf saliency model.
        num_classes (int): the number of classes
        label (tensor): label information.
        tau (float): threshold for confidence area
        lam (float): blending ratio between foreground map and background map
        intermediate (bool): if True return all the intermediates, if not return only loss.
    Shape:
        cam (N, C, H', W') where N is the batch size and C is the number of classes.
        saliency (N, 1, H, W)
        label (N, C)
    """
    
    b, c, h, w = cam.size()
    # pdb.set_trace()
    saliency = F.interpolate(saliency, size=(h, w))

    label_map = label.view(b, num_classes, 1, 1).expand(size=(b, num_classes, h, w)).bool()

    # Map selection
    label_map_fg = torch.zeros(size=(b, num_classes+1, h, w)).bool().cuda()
    label_map_bg = torch.zeros(size=(b, num_classes+1, h, w)).bool().cuda()

    label_map_bg[:, num_classes] = True
    label_map_fg[:, :-1] = label_map.clone()

    sal_pred = F.softmax(cam, dim=1)
#交集/显著性图（这会保留大面积占比的目标）（获取前景与显著性图交集>tau的类别）
    # pdb.set_trace()
    iou_saliency = (torch.round(sal_pred[:, :-1].detach()) * torch.round(saliency)).view(b, num_classes, -1).sum(-1) / \
                   (torch.round(sal_pred[:, :-1].detach()) + 1e-04).view(b, num_classes, -1).sum(-1)
#阈值化得到
    valid_channel = (iou_saliency > tau).view(b, num_classes, 1, 1).expand(size=(b, num_classes, h, w))
#大面积前景使用label剔除无关类
    label_fg_valid = label_map & valid_channel

    label_map_fg[:, :-1] = label_fg_valid
    label_map_bg[:, :-1] = label_map & (~valid_channel)

    # Saliency loss
    fg_map = torch.zeros_like(sal_pred).cuda()
    bg_map = torch.zeros_like(sal_pred).cuda()
#这里只是显著性图能够提供的前景应该是怎么样，需要使用local_sp去激活非显著区域
    fg_map[label_map_fg] = sal_pred[label_map_fg]#保留了前景各通道的信息
    bg_map[label_map_bg] = sal_pred[label_map_bg]

    fg_map_ = torch.sum(fg_map, dim=1, keepdim=True)
    bg_map = torch.sum(bg_map, dim=1, keepdim=True)

    bg_map = torch.sub(1, bg_map)
    sal_pred = fg_map_ * lam + bg_map * (1 - lam)

    loss = F.mse_loss(sal_pred, saliency)

    if intermediate:
        return loss, fg_map, bg_map, sal_pred
    else:
        return loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """cross_entropy. The wrapper function for :func:`F.cross_entropy`

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
            Default: None.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.
        ignore_index (int): Specifies a target value that is ignored and
            does not contribute to the input gradients. When
            ``avg_non_ignore `` is ``True``, and the ``reduction`` is
            ``''mean''``, the loss is averaged over non-ignored targets.
            Defaults: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)
    # apply weights and do the reduction
    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and reduction == 'mean':
        if class_weight is None:
            if avg_non_ignore:
                avg_factor = label.numel() - (label
                                              == ignore_index).sum().item()
            else:
                avg_factor = label.numel()

        else:
            # the average factor should take the class weights into account
            label_weights = torch.tensor([class_weight[cls] for cls in label],
                                         device=class_weight.device)
            if avg_non_ignore:
                label_weights[label == ignore_index] = 0
            avg_factor = label_weights.sum()

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()

    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights = bin_label_weights * valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         avg_non_ignore=False,
                         **kwargs):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
            Note: In bce loss, label < 0 is invalid.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int): The label index to be ignored. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.size(1) == 1:
        # For binary class segmentation, the shape of pred is
        # [N, 1, H, W] and that of label is [N, H, W].
        # As the ignore_index often set as 255, so the
        # binary class label check should mask out
        # ignore_index
        assert label[label != ignore_index].max() <= 1, \
            'For pred with shape [N, 1, H, W], its label must have at ' \
            'most 2 classes'
        pred = pred.squeeze(1)
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        # `weight` returned from `_expand_onehot_labels`
        # has been treated for valid (non-ignore) pixels
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.shape, ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask
    # average loss over non-ignored and valid elements
    if reduction == 'mean' and avg_factor is None and avg_non_ignore:
        avg_factor = valid_mask.sum().item()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None,
                       **kwargs):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]

def wsss_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         avg_non_ignore=False,
                         **kwargs):
    """
    Calculate the binary CrossEntropy loss.
    """
    #
    label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.shape, ignore_index)

    b,c,h,w=pred.size()
    pred = F.avg_pool2d(pred, kernel_size=(h, w), padding=0)

    label = torch.sum(label.view(b, c, -1),dim=-1)
    label = torch.where(label>0,torch.ones_like(label),torch.zeros_like(label)).unsqueeze(-1).unsqueeze(-1)

    return F.multilabel_soft_margin_loss(pred, label)


def depth_loss(pred,
                    target,
                    weight=None,
                    reduction='mean',
                    avg_factor=None,
                    class_weight=None,
                    ignore_index=-100,
                    avg_non_ignore=False,
                    **kwargs):
    """
    Calculate the binary CrossEntropy loss.
    """
    #
    label=target[0]
    sal=target[1]
    # pdb.set_trace()
    # sal=torch.mean(sal,dim=1,keepdim=True)
    pred_=target[2]
    depth_map=target[3]
    # b,c,h,w=pred.size()
    # pdb.set_trace()
    # label, weight, valid_mask = _expand_onehot_labels(
    #         label, weight, pred.shape, ignore_index)
    # # pdb.set_trace()
    # # b,c,h,w=pred_.size()
    # # pred_ = F.avg_pool2d(pred_, kernel_size=(h, w), padding=0)
    
    # b,c,h,w=pred.size()
    # label = torch.sum(label.view(b, c, -1),dim=-1)
    # label[label>0]=1.
    # label=label.float()
    # # label = label.unsqueeze(-1).unsqueeze(-1)
    # # pdb.set_trace()
    # # pred_=torch.roll(target[-1], shifts=-1, dims=1)#[bg,classes ch,..] [bs,c,h,w]
    # eps_loss =  eps_lossfn(pred_, sal, c-1, label[:,1:],
    #                     0.4, 0.5, intermediate=False)
    
    # pred=torch.roll(pred, shifts=-1, dims=0)

    depth_loss=torch.tensor(0.).cuda()


    return depth_loss

@MODELS.register_module()
class DepthLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_depth',
                 wsss=False,
                 eps_wsss=False,
                 depth=False,
                 avg_non_ignore=False):
        super().__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.wsss = wsss
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps_wsss = eps_wsss
        self.depth = depth
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        elif self.wsss:
            self.cls_criterion = wsss_cross_entropy
        elif self.eps_wsss:
            self.cls_criterion = eps_wsss_loss
        elif self.depth:
            self.cls_criterion = depth_loss
        else:
            self.cls_criterion = cross_entropy
        
        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        # Note: for BCE loss, label < 0 is invalid.
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
            ignore_index=ignore_index,
            **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
