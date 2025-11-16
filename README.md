# Introduction:

This is another stronger-centernet implemented without mmdetection. It removes the complex nesting in mmdetection, striving to achieve complete functionality with the simplest code. This implementation does not require installing any libraries that need to be compiled.

# Main libs:
```
torch==1.8.1+cu111
mmcv-full==1.4.2
```

# Run commands:

## train:

```
python -m torch.distributed.launch --nproc_per_node 2 train.py
```

## eval:

```
python test.py
```

## detect:

```
python demo.py
```

# Main results

| backbone | epoch | size    | mAP  |
| -------- | ----- | ------- | ---- |
| resnet18 | 1X    | 768*768 | 33.8 |

# Acknowledgement

This project is mainly implemented based on [ttfnet](https://github.com/ZJULearning/ttfnet), [mmdetection](https://github.com/open-mmlab/mmdetection), [CenterNet](https://github.com/xingyizhou/CenterNet), etc. Many Thanks for these repos.
