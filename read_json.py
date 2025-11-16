import json

path = './detect_results/coco/results.json'

with open(path,'r') as f:
    load_dict = json.load(f)
print(type(load_dict))
print(len(load_dict))
print(load_dict[0])
print(load_dict[1])
print(load_dict[2])
print(load_dict[3])
print(load_dict[99])
print(load_dict[100])

#print(load_dict)