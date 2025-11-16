
import torch.utils.data as data
import torch
import numpy as np
import json
import cv2
import math
from pycocotools.coco import COCO
from datasets_utils import *
from dataset_aug import MinIoURandomCrop,Expand,PhotoMetricDistortion,Resize


class certernet_datasets(data.Dataset):
    def __init__(self, mode = 'train',datasets = 'pascal',data_aug = False, input_size = 768): 
        super(certernet_datasets, self).__init__()
        self.input_h = input_size
        self.input_w = input_size
        self.img_scale=(self.input_w,self.input_h)
        self.aug = data_aug
        if self.aug == True:
            self.aug_rand_crop = MinIoURandomCrop()
            self.aug_rand_expand = Expand()
            self.aug_rand_color = PhotoMetricDistortion()
        self.aug_resize = Resize(self.img_scale, keep_ratio=False)
        self.mode = mode
        self.down_ratio = 8
        self.alpha = 0.54
        self.datasets = datasets
        # imagenet
        self.mean = np.array([123.675, 116.28 , 103.53 ],dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([58.395, 57.12 , 57.375],dtype=np.float32).reshape(1, 1, 3)
        if self.datasets == 'coco':
            self.num_classes = 80
            self.train_json_path = '/home/yanlb/work_space/dataset/coco2017/coco/annotations/instances_train2017.json' # new_small_object_instances_train2017.json         instances_train2017.json
            self.val_json_path = '/home/yanlb/work_space/dataset/coco2017/coco/annotations/instances_val2017.json'     # new_small_object_instances_val2017.json         instances_val2017.json
            self.train_image_path = '/home/yanlb/work_space/dataset/coco2017/coco/train2017/'
            self.val_image_path = '/home/yanlb/work_space/dataset/coco2017/coco/val2017/'
            self._valid_ids = [
              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
              24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
              37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
              58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
              72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
              82, 84, 85, 86, 87, 88, 89, 90]
            self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        else:
            self.num_classes = 20 
            self.train_json_path = './data/voc_2007/annotations/pascal_trainval0712.json'
            self.val_json_path = './data/voc_2007/annotations/pascal_test2007.json'
            self.train_image_path = './data/voc_2007/images/'
            self.val_image_path = './data/voc_2007/images/'
            self._valid_ids = np.arange(1, 21, dtype=np.int32)
            self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        if self.mode == 'train':
            self.coco = COCO(self.train_json_path)
            self.image_path = self.train_image_path
        else:
            self.coco = COCO(self.val_json_path)
            self.image_path = self.val_image_path
        
        #self.img_id_list =self.coco.getImgIds() # 所有图像的id

        '''
        =================================================================
        a more elegant solution for remove images that do not contain GT:
        cleaning data, thereby need not use my_collate
        =================================================================
        
        self.img_id_list_ =self.coco.getImgIds() # 所有图像的id
        if self.mode == 'train':
            self.img_id_list = []
            for i in range(len(self.img_id_list_)):
                img_id = self.img_id_list_[i]
                ann_ids = self.coco.getAnnIds(imgIds=[img_id])
                anns = self.coco.loadAnns(ids=ann_ids)
                if len(anns) != 0:
                    self.img_id_list.append(img_id)
        else:
            self.img_id_list = self.img_id_list_

        '''
        self.img_id_list_ =self.coco.getImgIds() # 所有图像的id
        if self.mode == 'train':
            self.img_id_list = []
            for i in range(len(self.img_id_list_)):
                img_id = self.img_id_list_[i]
                ann_ids = self.coco.getAnnIds(imgIds=[img_id])
                anns = self.coco.loadAnns(ids=ann_ids)
                if len(anns) != 0:
                    self.img_id_list.append(img_id)
        else:
            self.img_id_list = self.img_id_list_
    def __getitem__(self,index):
        img_id = self.img_id_list[index]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        #print(file_name)
        img = cv2.imread(self.image_path+file_name).astype(np.float32)
        height,width = img.shape[0],img.shape[1]  
        ann_ids = self.coco.getAnnIds(imgIds=[img_id],iscrowd = False)
        anns = self.coco.loadAnns(ids=ann_ids)
        bbox_num = len(anns)

        # NOTE:
        # there is a potential bug if batchsize is 1, dataloader iter will throw None.
        # or even though batchsize is 10, there is case when all images are bad, also throwing a bug.
        # best way is to clean data. 
        
        # if self.mode == 'train':
        #     if bbox_num == 0:
        #         return None

        scale_h,scale_w = self.input_h/height, self.input_w/width # only for val
        gt_bboxes = np.zeros((bbox_num,4),dtype = np.float32) # xyxy
        gt_labels = np.zeros((bbox_num,),dtype = np.int64)

        for k in range(bbox_num): # get descending order via area
          ann = anns[k] # xywh
          cls_id = int(self.cat_ids[ann['category_id']])
          gt_bboxes[k,0] = ann['bbox'][0]
          gt_bboxes[k,1] = ann['bbox'][1]
          gt_bboxes[k,2] = ann['bbox'][0] + ann['bbox'][2]
          gt_bboxes[k,3] = ann['bbox'][1] + ann['bbox'][3]
          gt_labels[k] = cls_id

        # data augmentation
        result = {'img':img, 'gt_bboxes':gt_bboxes, 'gt_labels':gt_labels}
        if self.aug == True:
            result = self.aug_rand_color(result)
            result = self.aug_rand_expand(result)
            result = self.aug_rand_crop(result)
        result = self.aug_resize(result)
        img,gt_bboxes,gt_labels = result['img'],result['gt_bboxes'],result['gt_labels']
        
        # random flip
        flipped = False
        if self.mode == 'train':
            if np.random.random() < 0.5: 
                flipped = True
                img = img[:, ::-1, :] 
                gt_bboxes[:,[0,2]] = img.shape[1] - gt_bboxes[:,[2,0]] - 1
        # print('-----------------')
        # show_bbox(img,gt_bboxes)
        gt_bboxes = torch.from_numpy(gt_bboxes)
        gt_labels = torch.from_numpy(gt_labels)
        boxes_areas_log = bbox_areas(gt_bboxes).log()
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))
        gt_bboxes = gt_bboxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        output_h = self.input_h // self.down_ratio
        output_w = self.input_w // self.down_ratio
        feat_gt_bboxes = gt_bboxes / self.down_ratio
        feat_gt_bboxes[:, [0, 2]] = torch.clamp(feat_gt_bboxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_bboxes[:, [1, 3]] = torch.clamp(feat_gt_bboxes[:, [1, 3]], min=0,
                                               max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_bboxes[:, 3] - feat_gt_bboxes[:, 1],
                            feat_gt_bboxes[:, 2] - feat_gt_bboxes[:, 0])

        ct_ints = (torch.stack([(gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2,
                                (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()


        heatmap_channel = self.num_classes
        heatmap = gt_bboxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_bboxes.new_zeros((output_h, output_w))
        box_target = gt_bboxes.new_ones((4, output_h, output_w)) * -1
        reg_weight = gt_bboxes.new_zeros((1, output_h, output_w))

        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k]
            fake_heatmap = fake_heatmap.zero_()
            draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item())
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)
            # cv2.imshow('',heatmap[cls_id].numpy()/np.max(heatmap[cls_id].numpy()))
            # cv2.waitKey(0)
            box_target_inds = fake_heatmap > 0
            box_target[:, box_target_inds] = gt_bboxes[k][:, None] # xyxy of bbox in input image 
            cls_id = 0
            local_heatmap = fake_heatmap[box_target_inds]
            ct_div = local_heatmap.sum()
            local_heatmap *= boxes_area_topk_log[k]
            reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
        # print('^^^^^^^^^^^^^^show box_target image^^^^^^^^^^^^^^^^^^')
        # cv2.imshow('',reg_weight[0].numpy()*100/np.max(reg_weight[0].numpy()))
        # cv2.waitKey(0)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = (img - self.mean) / self.std 
        img = img.transpose(2, 0, 1) 
        ret = {'input': torch.from_numpy(img), 'hm': heatmap,\
              'box_target':box_target,'reg_weight':reg_weight}
        if not self.mode == 'train': 
          meta = {'hw':(height,width),'scale_hw':(scale_h,scale_w), 'img_id': img_id}
          ret['meta'] = meta
        return ret
    def __len__(self):
        return len(self.img_id_list)
def my_collate(batch):
    batch = filter(lambda x : x is not None, batch)
    return data.dataloader.default_collate(list(batch))


#===============
# test
#===============

if __name__ == '__main__':

    datasets = certernet_datasets(mode = 'val',datasets = 'coco',data_aug = True)
    # for i in range(len(datasets)):
    #     print(i)
    #     ret = datasets.__getitem__(i)
    #     num = datasets.__len__()
    #     #print(ret['meta']['img_id'])


    ret = datasets.__getitem__(100)
    num = datasets.__len__()
    print(ret['meta']['img_id'])


