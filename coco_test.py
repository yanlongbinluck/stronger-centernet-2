
import cv2
from pycocotools.coco import COCO 
import matplotlib.pyplot as plt

def show_bbox(image_cv2,anns):
    for i in range(len(anns)):
        x,y,w,h = anns[i]['bbox'] 
        #print('area',anns[i]['area'])
        #print('bbox_area',w*h)
        x1 = int(x)
        y1 = int(y)
        x2 = int(x+w)
        y2 = int(y+h)
        cv2.rectangle(image_cv2,(x1,y1),(x2,y2),(0,0,255))
    cv2.imshow('',image_cv2)
    cv2.waitKey(1000)

annFile = '/home/yanlb/work_space/dataset/coco2017/coco/annotations/instances_val2017.json'
root = '/home/yanlb/work_space/dataset/coco2017/coco/val2017/'
coco = COCO(annFile) 


cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories:\n{}\n'.format(' '.join(nms)))
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories:\n{}'.format(' '.join(nms)))


catIds=coco.getCatIds(catNms=['person','dog']) 


imgIds = coco.getImgIds(catIds=catIds)


img =coco.loadImgs(imgIds[0])[0] 
file_name = img['file_name'] 
annIds = coco.getAnnIds(imgIds=img['id']) 
anns = coco.loadAnns(annIds) 


I = cv2.imread(root + file_name) 
plt.axis('off')
plt.imshow(I[:,:,::-1])


show_bbox(I,anns)


coco.showAnns(anns)
plt.show()