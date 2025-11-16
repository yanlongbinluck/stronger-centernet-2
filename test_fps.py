
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

#===============
# pascal or coco
#===============
datasets = 'coco'
backbone = 'resnet18' # 'darknet53' 'resnet18' 'resnet34' 'resnet50' 'swin_transformer'
affm = False
ddh = False
input_size = 768
ttfnet_official_model_format = False
threshold = 0.2 # 0.2 or None. Default is None, because there is already a threshold(0.001) in soft_nms


if datasets == 'coco':
    save_dir='./detect_results/coco'
    save_results = save_results_coco
    run_eval = run_eval_coco
    classes = 80
else:
    save_dir='./detect_results/pascal'
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



    device = 'cuda:0'
    model = model.to(device)
    model.eval()
    total_time = 0
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
        total_time = total_time + (end - start)
        print('precessing No. {} | inference time : {}s.'.format(i,end - start))
    print('Average FPS is:',1/(total_time/len(val_data_loader)))
        

if __name__ == '__main__':
    main()
