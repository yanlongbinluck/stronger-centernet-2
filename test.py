
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
torch.set_printoptions(threshold = 10000000)
from model import Stronger_CenterNet
from torch.utils.data import DataLoader
from datasets import certernet_datasets
import time
from test_utils import run,run_eval_coco,save_results_coco,run_eval_pascal,save_results_pascal
import numpy as np
import cv2
from common_utils import load_model,save_model,write_txt

coco_name = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
              'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
              'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
              'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
              'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def show_bbox_form_dict_result(img_id,result):
    img = cv2.imread('/home/yanlb/work_space/dataset/coco2017/coco/val2017/%012d.jpg'%img_id)
    dict_result = result # result is dict {1: array(N*5), 2:array(N*5),... 80:array(N*5)}
    for i in range(80):
        dets = dict_result[i+1] # N*5 array, key of dict is int
        if dets.size != 0:
            for j in range(len(dets)):
                gt_det = dets[j]
                x1 = int(gt_det[0])
                y1 = int(gt_det[1])
                x2 = int(gt_det[2])
                y2 = int(gt_det[3])
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2) # BGR
                cv2.putText(img,coco_name[i],(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,150),1)
    cv2.imshow('',img)
    cv2.waitKey(0)


# from torchvision import transforms
# from PIL import Image
# import cv2

#===============
# pascal or coco
#===============
datasets = 'coco'
backbone = 'resnet18' # 'darknet53' 'resnet18' 'resnet34' 'resnet50' 'swin_transformer'
affm =True
ddh = True
input_size = 768
ttfnet_official_model_format = False
threshold = None # 0.2 or None.

model_path = "./train_log/2025-11-15-14-06-17/model_last.pth"

if datasets == 'coco':
    save_dir='./eval_results_json'
    save_results = save_results_coco
    run_eval = run_eval_coco
    classes = 80
else:
    save_dir='./eval_results_json'
    save_results = save_results_pascal
    run_eval = run_eval_pascal
    classes = 20

def main():

    val_data = certernet_datasets(mode = 'val',datasets = datasets, data_aug = False,input_size = input_size)
    print('there are {} val images'.format(len(val_data)))
    val_data_loader = DataLoader(dataset=val_data,
                                   num_workers=0,
                                   batch_size=1,
                                   shuffle=False,
                                   pin_memory=True)

    
    model = Stronger_CenterNet(backbone = backbone, affm = affm, ddh = ddh)

    if ttfnet_official_model_format == True:
    # this is for ----> ttfnet53_1x-4811e4.pth
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'],strict=True) 
    else:
    # This is for centernet format 
        model = load_model(model,model_path)

    device = 'cuda:0'
    model = model.to(device)
    model.eval()
    results = {}
    for i ,batch in enumerate(val_data_loader):
        images, meta = {},{}
        meta_data = batch['meta']
        img_id = meta_data['img_id']
        del meta_data['img_id']
        images[1.0] = batch['input']
        meta[1.0] = meta_data
        pre_processed_images = {'images':images,'meta':meta}
        torch.cuda.synchronize()
        start = time.time()
        ret = run(pre_processed_images,model,threshold)
        torch.cuda.synchronize()
        end = time.time()
        print('precessing No. {} | inference time : {}s.'.format(i,end - start))   
        results[img_id.numpy().astype(np.int32)[0]] = ret['results']
        #show_bbox_form_dict_result(img_id.numpy().astype(np.int32)[0],ret['results'])
    save_results(results, save_dir=save_dir)
    run_eval(save_dir=save_dir)




if __name__ == '__main__':
    main()
