
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import cv2
import glob
import torch
import numpy as np
from datasets_utils import *
from demo_utils import *
import time
import os
from common_utils import load_model,save_model,write_txt
from model import Stronger_CenterNet


coco_name = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', # 5
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', # 11
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', # 18
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

#=============================================================================================
image_path = './input_image'
model_path = "./train_log/2025-11-15-14-06-17/model_last.pth"
dataset = 'coco'
threshold = 0.2 # 0.2
K = 500
ttfnet_official_model_format = False
input_h,input_w = 768,768
down_ratio = 8
affm =True
ddh = True
backbone ='resnet18'
#=============================================================================================




if dataset == 'coco':
    mean = np.array([123.675, 116.28 , 103.53 ],dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([58.395, 57.12 , 57.375],dtype=np.float32).reshape(1, 1, 3)

image_list = sorted(glob.glob(image_path+'/*.*'))


def main():
    model = Stronger_CenterNet(backbone = backbone,affm = affm,ddh = ddh)
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

    for i in range(len(image_list)):
        start_time = time.time()
        img = cv2.imread(image_list[i])
        file_name = os.path.basename(image_list[i])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        height,width = img.shape[0],img.shape[1]
        
        scale_h,scale_w = input_h/height, input_w/width  

        inp = cv2.resize(img,(input_w,input_h),interpolation=cv2.INTER_LINEAR)
        
        inp = inp.astype(np.float32)
        inp = (inp - mean)/std

        meta = {'hw':(height,width),'scale_hw':(scale_h,scale_w)}
        inp = torch.from_numpy(inp)
        inp = inp.permute(2,0,1).unsqueeze(0).to(device) # from [1,512,512,3] to [1, 3, 512, 512]
        dets = process(inp, model, K) # dets: [1, 100, 6] top100个 xyxy store class_id
        mask = dets[0,:,4].ge(threshold)
        dets = dets[:,mask,:]
        dets = post_process(dets, meta)
        #print(dets)


        # plot bboxes
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        print(len(dets))
        if len(dets) > 0:
          for j in range(len(dets)):
              gt_det = dets[j]
              x1 = int(gt_det[0])
              y1 = int(gt_det[1])
              x2 = int(gt_det[2])
              y2 = int(gt_det[3])
              cv2.putText(img,coco_name[int(gt_det[5])]+" %.2f"%(gt_det[4].cpu().numpy()),(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1) # BGR
              draw = cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),1)
        else:
          draw = img
        end_time = time.time()
        print(end_time - start_time)
        cv2.imwrite('./detect_results_image/{}'.format(file_name),draw)



if __name__ == '__main__':
    main()