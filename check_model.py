
import torch
model_path = './weights/coco/ttfnet53_1x-4811e4.pth'
checkpoint = torch.load(model_path)

i = 1
for name,value in checkpoint['state_dict'].items():
    i = i + 1
    print(i)
    print(name,value.size())