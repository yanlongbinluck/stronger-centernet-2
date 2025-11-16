

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['MASTER_PORT'] = '20000'
import torch
from torch.nn.utils import clip_grad
from model import Stronger_CenterNet
from loss import CtdetLoss,AverageMeter
from torch.utils.data import DataLoader
from datasets import certernet_datasets,my_collate
import time
import random
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda import amp
from common_utils import load_model,save_model,write_txt

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
total_gpu_number = torch.cuda.device_count()
print("total gpu number:", total_gpu_number)

#          Config
# ================================================================================================
datasets = 'coco' 
backbone ='resnet18' # 'darknet53' 'resnet18' 'resnet34' 'resnet50' 'res2net101'
affm = True
ddh = True
grad_max_norm = 35 # grad clip. default is 35.
optim = 'SGD' # Adam or SGD or AdamW
worker_per_gpu = 4 # mmdetection default is 2
total_batch_size = 128
batch_size = 32 # per gpu.  darknet 12,lr 0.015; resnet 16,lr0.016
gradient_acc_step = total_batch_size//(batch_size*total_gpu_number) # total size = batch_size * gradient_acc_step
resume = False
load_from_1x_model = False
val_in_train_processing = False
print_interval = 10
start_epoch = 0 # epoch id is 0-(end_epoch - 1)
current_iter = 0
init_lr = 0.016
train_schedule = '1x' # 1x 2x 10x
input_size = 768
seed = 888
amp_use = True
warmup = True
warmup_iters = 1000 # about 0.5 epoch for warmup
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# ================================================================================================

if train_schedule == '1x':
    epoch_times = 1
    augmentation = False
elif train_schedule == '10x':
    epoch_times = 10
    augmentation = True

end_epoch = 12*epoch_times
lr_step=[9*epoch_times,11*epoch_times] # epoch id

def mkdir(path):
    path=path.strip() 
    path=path.rstrip("\\") 
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path+' has been created')

def main():
    setup_seed(seed + local_rank)
    print('rank:',local_rank,'seed:',seed + local_rank)

    best=1e10
    global start_epoch # this global is to let model from n epoch start to train
    global current_iter
    criterion = CtdetLoss().to(local_rank)
    model = Stronger_CenterNet(backbone = backbone,affm = affm,ddh = ddh).to(local_rank)

    optimizer = get_optimizer(model,init_lr,opt = optim)

    # train_data_loader
    train_data = certernet_datasets(mode = 'train',datasets = datasets,data_aug = augmentation,input_size = input_size)
    print('there are {} train images'.format(len(train_data)))
    train_sampler = DistributedSampler(train_data)
    train_data_loader = DataLoader(dataset=train_data,
                                   num_workers=worker_per_gpu,
                                   batch_size=batch_size,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   #collate_fn = my_collate
                                   )

    # val_data_loader
    val_data = certernet_datasets(mode = 'val',datasets = datasets,data_aug = False, input_size = input_size)
    print('there are {} val images'.format(len(val_data)))
    val_data_loader = DataLoader(dataset=val_data,
                                   num_workers=worker_per_gpu,
                                   batch_size=batch_size, 
                                   shuffle=False
                                   )

    # resume
    if resume == True:
        checkpoint_model_path = save_dir + '/model_last.pth'
        model, optimizer, start_epoch = load_model(model, checkpoint_model_path, optimizer, resume, init_lr, lr_step)
        current_iter = start_epoch * len(train_data)
        print('=======================')
        print('resume from {} epoch, total {} iter.'.format(start_epoch, current_iter))
        print('=======================')
        
        # load params of optimizer to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=local_rank, non_blocking = True)

    if load_from_1x_model == True:
        model = load_model(model, model_path = "./model_last.pth")
        print("load weights from 1x model!")

    if torch.cuda.device_count() > 1:
            print("using GPU{}!".format(local_rank))
            model = DDP(model, device_ids=[local_rank])
    if amp_use == True:
        scaler = amp.GradScaler(enabled=True)
    else:
        scaler = None

    lr_group = get_regular_lr(optimizer)
    for epoch in range(start_epoch,end_epoch):
        train_sampler.set_epoch(epoch)
        adjust_lr_epoch(optimizer,epoch,init_lr,lr_step)
        start_time = time.time()
        train_epoch(train_data_loader,model,optimizer,lr_group,criterion,epoch,local_rank,scaler)
        end_time = time.time()
        ELA_time = (end_time - start_time)
        ELA_time = time.strftime('%H:%M:%S',time.gmtime(ELA_time))
        
      
        # save checkpoint
        if  local_rank == 0:
            save_model(save_dir + '/model_last.pth',epoch,model,optimizer)
            print('Time/epoch:',ELA_time)



def train_epoch(train_data_loader,model,optimizer,lr_group,criterion,epoch,local_rank,scaler):
    global current_iter # when resume, it to let model know what is current iter now
    losses = AverageMeter()
    model.train()

    for i ,batch in enumerate(train_data_loader):
        if optim == 'SGD':
            adjust_lr_warmup(optimizer,current_iter,lr_group)
        current_base_lr = get_regular_lr(optimizer)[0]
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].to(local_rank)

        if amp_use == True:
            with amp.autocast(enabled=True):
                output = model(batch['input'])
                loss_stats = criterion(output,batch)
            loss = loss_stats['loss'] / gradient_acc_step
            loss_hm = loss_stats['hm_loss']
            loss_wh = loss_stats['wh_loss']
            scaler.scale(loss).backward()
            if (i+1) % gradient_acc_step == 0:
                scaler.unscale_(optimizer) # this is used for grad clip in normal mode
                clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=grad_max_norm, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            output = model(batch['input'])
            loss_stats = criterion(output,batch)
            loss = loss_stats['loss'] / gradient_acc_step
            loss_hm = loss_stats['hm_loss']
            loss_wh = loss_stats['wh_loss']
            loss.backward()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=grad_max_norm, norm_type=2)
            if (i+1) % gradient_acc_step == 0:
                optimizer.step()
                optimizer.zero_grad()
        losses.update(loss.item() * gradient_acc_step) # for log
        if local_rank == 0 and i % print_interval == 0:
            write_txt({'[epoch:{},{}/{}]'.format(epoch,i,len(train_data_loader)):'',\
                       'lr':'%.6f'%current_base_lr,\
                       'total loss':'%.4f'%losses.current,\
                       'average_loss':'%.4f'%losses.avg,\
                       'hm_loss':'%.4f'%loss_hm.item(),\
                       'wh_loss':'%.4f'%loss_wh.item()},\
                        save_dir+'/train_log.txt')
            print('[epoch:{},{}/{}]'.format(epoch,i,len(train_data_loader)),\
                'lr:%.6f'% current_base_lr,\
                'total_loss:%.4f'% losses.current,\
                'average_loss:%.4f' % losses.avg,\
                'hm_loss:%.4f'% loss_hm.item(),\
                'wh_loss:%.4f'% loss_wh.item())
        current_iter = current_iter + 1

def val_epoch(val_data_loader,model,criterion,local_rank):

    losses = AverageMeter()
    model = model.module
    model.eval()
    for i ,batch in enumerate(val_data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].to(local_rank, non_blocking=True)
        output = model(batch['input'])
        loss_stats = criterion(output,batch)
        loss = loss_stats['loss']
        losses.update(loss.item())
    return losses.avg


def get_regular_lr(optimizer):
    lr_group=[] # get  all lr for every group
    for param_group in optimizer.param_groups:
        lr_group += [param_group['lr']]
    return lr_group

def adjust_lr_epoch(optimizer,epoch,init_lr,lr_step):
    if epoch in lr_step: # if need reduce the 10 times
        lr_group = get_regular_lr(optimizer)
        lr_group = [_lr * 0.1 for _lr in lr_group]
        for param_group, lr in zip(optimizer.param_groups, lr_group):
            param_group['lr'] = lr

def adjust_lr_warmup(optimizer,current_iter,lr_group,warmup_iters=warmup_iters,warmup='linear',warmup_radio=0.2):
    if current_iter < warmup_iters: # if iter < 500, use warmup_lr. from 0 to 499
        k = (1 - current_iter / warmup_iters) * (1 - warmup_radio)
        lr_group = [_lr * (1-k) for _lr in lr_group]
        for param_group, lr in zip(optimizer.param_groups, lr_group):
            param_group['lr'] = lr

def get_optimizer(model,init_lr,opt = 'SGD',weight_decay=0.0004):

    if opt == 'AdamW': # for swin backbone
        optimizer = torch.optim.AdamW(model.parameters(),lr = init_lr,betas=(0.9,0.999), weight_decay=0.05)
        print('using AdamW optimizer!')
        return optimizer

    params = []
    for key,value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':init_lr*2,'weight_decay':0}]
            else:
                params += [{'params':[value],'lr':init_lr,'weight_decay':weight_decay}]
    if opt == 'SGD':
        optimizer = torch.optim.SGD(params,momentum=0.9)
        print('using SGD optimizer!')
    if opt == 'Adam':
        optimizer = torch.optim.Adam(params)
        print('using Adam optimizer!')
    return optimizer



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    


if __name__ == '__main__':
    save_dir = "./train_log/{}".format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    if local_rank == 0:
        mkdir(save_dir)
    main()
