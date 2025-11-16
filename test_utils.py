

import torch
import torch.nn as nn
import numpy as np


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

#===============
# pascal or coco
#===============
dataset = 'coco'
down_ratio = 8

if dataset == 'coco':
    num_classes = 80
    annot_path = '/home/yanlb/work_space/dataset/coco2017/coco/annotations/instances_val2017.json'
    coco = COCO(annot_path)
    _valid_ids = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
      24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
      58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
      72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
      82, 84, 85, 86, 87, 88, 89, 90]
else:
    num_classes = 20
    annot_path = './data/voc/annotations/pascal_test2007.json'
    coco = COCO(annot_path)
    img_id_list =coco.getImgIds()
    images = sorted(img_id_list)

max_per_image = 100 

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

def ctdet_decode(heat, wh, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)  
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    xs = xs.view(batch, K, 1) * down_ratio
    ys = ys.view(batch, K, 1) * down_ratio
    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 4) # four distences from center point to edge in raw image
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1], 
                        ys - wh[..., 1:2],
                        xs + wh[..., 2:3] + 1, # TODO add 1, mAP from 32 to 33.0, but offical is 33.1
                        ys + wh[..., 3:4] + 1], dim=2) # return xyxy of predicted bbox
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections



def process(images,model):
    with torch.no_grad():
      hm,wh = model(images)
      hm = hm.sigmoid_() #[1, 20, 128, 128]
      dets = ctdet_decode(hm, wh, K=100)
    return dets




def ctdet_post_process(dets,meta,num_classes):

  # dets: batch x max_dets x dim
  # return 1-based class det dict


  dets[:,:, [0,2]] = dets[:,:,  [0,2]] / meta['scale_hw'][1].float() # from 512*512 to raw height*width
  dets[:,:, [1,3]] = dets[:,:,  [1,3]] / meta['scale_hw'][0].float()
  dets[:,:,  [0,2]] = dets[:,:,  [0,2]].clamp(min=0, max=meta['hw'][1].item() - 1) # limit in raw image
  dets[:,:,  [1,3]] = dets[:,:, [1,3]].clamp(min=0, max=meta['hw'][0].item() - 1)
  dets = dets.numpy()
  #print(dets) # TODO
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def post_process(dets, meta, scale=1): 

    dets = dets.detach().cpu()
    dets = dets.view(1, -1, dets.shape[2]) # to (1, 100, 6)
    dets = ctdet_post_process(dets.clone(), meta, num_classes)
    

    for j in range(1, num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

def merge_outputs(detections):
    scales = [1.0] 
    results = {}
    for j in range(1, num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(scales) > 1:
         print("multi scales test mode needs nms, please install.")
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
      kth = len(scores) - max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

def run(image_or_path_or_tensor, model, threshold):
    scale = 1.0
    pre_processed_images = image_or_path_or_tensor
    detections = []
    images = pre_processed_images['images'][scale]
    meta = pre_processed_images['meta'][scale]
    images = images.cuda()
    dets = process(images, model)

    if threshold is not None:
        mask = dets[0,:,4].ge(threshold)
        dets = dets[:,mask,:]

    dets = post_process(dets, meta, scale)
    detections.append(dets)
    results = merge_outputs(detections)


    return {'results': results}

def _to_float(x):
    return float("{:.2f}".format(x))


def convert_eval_format_pascal(all_bboxes):
    detections = [[[] for __ in range(len(img_id_list))] for _ in range(num_classes + 1)]
    for i in range(len(img_id_list)):
      img_id = images[i]
      for j in range(1, num_classes + 1):
        if isinstance(all_bboxes[img_id][j], np.ndarray):
          detections[j][i] = all_bboxes[img_id][j].tolist()
        else:
          detections[j][i] = all_bboxes[img_id][j]
    return detections

def convert_eval_format_coco(all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = _valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(_to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(_to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections


def save_results_coco(results, save_dir):
    json.dump(convert_eval_format_coco(results), open('{}/results.json'.format(save_dir), 'w'))

def save_results_pascal(results, save_dir):
    json.dump(convert_eval_format_pascal(results), open('{}/results.json'.format(save_dir), 'w'))


def run_eval_coco(save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    coco_dets = coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def run_eval_pascal(save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    os.system('/home/yanlb/anaconda3/bin/python tools/reval.py ' + '{}/results.json'.format(save_dir))

#===================================================================================
# when generate results.json, you can directly run run_eval_pascal() as follow:
#===================================================================================

#run_eval_pascal('./detect_results/pascal')
