from pycocotools.coco import COCO


train_json_path = './data/coco/annotations/instances_train2017.json'
coco = COCO(train_json_path)
img_id = 453566
ann_ids = coco.getAnnIds(imgIds=[img_id])
print(ann_ids)
anns = coco.loadAnns(ids=ann_ids)
print(anns)