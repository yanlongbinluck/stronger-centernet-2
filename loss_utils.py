
import torch
import torch.nn as nn
import torch.nn.functional as F

def new_sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def ct_focal_loss(pred, gt, gamma=2.0):
    """
    Focal loss used in CornerNet & CenterNet. Note that the values in gt (label) are in [0, 1] since
    gaussian is used to reduce the punishment and we treat [0, 1) as neg example.

    Args:
        pred: tensor, any shape.
        gt: tensor, same as pred.
        gamma: gamma in focal loss.

    Returns:

    """
    # print('-----------')
    # print(torch.min(gt),torch.max(gt))
    # print(torch.min(pred),torch.max(pred))

    pos_inds = gt.eq(1).float() # batch*C*128*128
    neg_inds = gt.lt(1).float()

    # print('****************')
    # print(torch.min(torch.log(pred)),torch.max(torch.log(pred)))
    # print(torch.min(torch.pow(1 - pred, gamma)),torch.max(torch.pow(1 - pred, gamma)))
    neg_weights = torch.pow(1 - gt, 4)  # reduce punishment
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum() # it means number of GTs of batch image
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    # print(pos_loss,neg_loss)

    if num_pos == 0:
        return neg_loss
    return (pos_loss + neg_loss) / num_pos


class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = ct_focal_loss
  def forward(self, out, target):
    return self.neg_loss(out, target)

def giou_loss(pred,
              target,
              weight,
              avg_factor):
    """GIoU loss.
    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    pred,target are xyxy
    """
    assert avg_factor is not None
    pos_mask = weight > 0
    weight = weight[pos_mask].float()


    bboxes1 = pred[pos_mask].view(-1, 4) # from [batch,128,128,4] to [batch*128*128,4]
    bboxes2 = target[pos_mask].view(-1, 4)

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
    enclose_x1y1 = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    enclose_x2y2 = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)

    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]  # i.e. C in paper
    u = ap + ag - overlap
    gious = ious - (enclose_area - u) / enclose_area
    iou_distances = 1 - gious
    return torch.sum(iou_distances * weight)[None] / avg_factor


class Reg_Loss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(Reg_Loss, self).__init__()
    self.wh_loss = giou_loss

  def forward(self, 
              pred,
              target,
              weight,
              avg_factor):
    return self.wh_loss(pred, target, weight, avg_factor)