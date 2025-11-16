
import torch
import torch.nn as nn
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
import matplotlib.pyplot as plt

#===============
# pascal or coco
#===============
num_classes = 80
down_ratio = 8



def ctdet_post_process(dets, meta, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict

    dets = dets[0]
    dets[:, [0,2]] = dets[:, [0,2]] / meta['scale_hw'][1] # from 512*512 to raw height*width
    dets[:, [1,3]] = dets[:, [1,3]] / meta['scale_hw'][0]
    dets[:, [0,2]] = dets[:, [0,2]].clamp(min=0, max=meta['hw'][1] - 1)
    dets[:, [1,3]] = dets[:, [1,3]].clamp(min=0, max=meta['hw'][0] - 1)
    return dets

def post_process(dets, meta): 

    dets = dets.reshape(1, -1, dets.shape[2]) # (1, 100, 6)
    dets = ctdet_post_process(dets.clone(), meta, num_classes)
    return dets

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _topk(scores, K):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, K):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)  
    scores, inds, clses, ys, xs = _topk(heat, K)
    xs = xs.view(batch, K, 1) * down_ratio
    ys = ys.view(batch, K, 1) * down_ratio
    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 4) # four distences from center point to edge in raw image
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1], 
                        ys - wh[..., 1:2],
                        xs + wh[..., 2:3], 
                        ys + wh[..., 3:4]], dim=2) # return xyxy of predicted bbox
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections



def process(images,model,K):
    with torch.no_grad():
      hm,wh = model(images)
      hm = hm.sigmoid_() #[1, 20, 128, 128]
      dets = ctdet_decode(hm, wh, K)
    return dets
