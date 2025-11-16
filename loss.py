
import torch
from loss_utils import *



class AverageMeter(object):
    # computer average loss
    def __init__(self, ):
        super(AverageMeter, self).__init__()
        self.sum = 0
        self.count = 0

    def update(self,current_loss):
        self.current = current_loss
        self.sum += current_loss
        self.count += 1
        self.avg = self.sum / self.count

class CtdetLoss(torch.nn.Module): 

  def __init__(self):
    super(CtdetLoss, self).__init__()
    self.crit = FocalLoss() 
    self.crit_reg = Reg_Loss()
    self.base_loc = None
    self.down_ratio = 8

  def forward(self, outputs, batch): 
    hm_loss, wh_loss = 0, 0
    # old centernet resdcn18 output:
    # outputs[0]['hm']
    # outputs[0]['wh']
    hm = outputs[0] # [batch, 20, 128, 128]
    wh = outputs[1] # [batch, 4, 128, 128]



    H, W = hm.shape[2:] # [batch,C,128,128]
    # print('3333333333333')
    # print(hm)
    hm = new_sigmoid(hm) 
    hm_loss += self.crit(hm, batch['hm'])  
    mask = batch['reg_weight'].view(-1, H, W) # from [batch,1,128,128] to [batch,128,128]
    avg_factor = mask.sum() + 1e-4 # 200~300
    if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
        base_step = self.down_ratio
        shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                dtype=torch.float32, device=hm.device) # 0,5,9,...509,128 point
        shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                dtype=torch.float32, device=hm.device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x) # [128, 128], [128, 128]
        self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, 128, 128)

    pred_boxes = torch.cat((self.base_loc - wh[:, [0, 1]],
                            self.base_loc + wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
    # center point - four distences from center point to edge in raw image
    # get absolute coordinate: 
    # pred_boxes: xyxy
    # boxes: xyxy

    boxes = batch['box_target'].permute(0, 2, 3, 1) # from [batch,4,128,128] to [batch,128,128,4]
    wh_loss += self.crit_reg(pred_boxes, boxes, mask, avg_factor) * 5

    #print(hm_loss,wh_loss)
    loss = hm_loss + wh_loss 
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,'wh_loss': wh_loss}
    return loss_stats