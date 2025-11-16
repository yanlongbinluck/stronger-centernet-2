import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2dPack as DCNv2 # DCNv2
import re
from resnet import resnet18,resnet34,resnet50,resnet101,resnet152
from res2net import Res2Net
from darknet import darknet53

class SPP(nn.Module): # size 2X, channel 1/2X
    def __init__(self,in_channel):
        super(SPP,self).__init__()
        self.weight = nn.Conv2d(in_channel*4, 4, 1, stride=1, padding=0, dilation=1)
    def forward(self,x):
        x1 = F.max_pool2d(x,3,stride=1,padding=1) # raw  is 5,9,13
        x2 = F.max_pool2d(x,5,stride=1,padding=2)
        x3 = F.max_pool2d(x,7,stride=1,padding=3)
        weight = self.weight(torch.cat((x,x1,x2,x3),dim=1))
        weight = F.softmax(weight,dim=1)
        x = x * weight[:,0:1,:,:] + x1 * weight[:,1:2,:,:] + x2 * weight[:,2:3,:,:] + x3 * weight[:,3:4,:,:]
        return x

class Fuse(nn.Module): # size 2X, channel 1/2X
    def __init__(self,in_channel):
        super(Fuse,self).__init__()
        self.weight = nn.Conv2d(in_channel*2, 2, 1, stride=1, padding=0, dilation=1)
    def forward(self,x1,x2):
        weight = self.weight(torch.cat((x1,x2),dim=1))
        weight = F.softmax(weight,dim=1)
        return x1 * weight[:,0:1,:,:] + x2 * weight[:,1:2,:,:]

class Fuse_two(nn.Module): # size 2X, channel 1/2X
    def __init__(self,in_channel):
        super(Fuse_two,self).__init__()
        self.mask = nn.Sequential(
                     nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, dilation=1,bias = True),
                     nn.Sigmoid()
                     )
    def forward(self,x1,x2):
        mask = self.mask(x2) # x1 is main flow
        return x1 * (1-mask) + x2 * mask

class Fuse_three(nn.Module): # size 2X, channel 1/2X
    def __init__(self,in_channel):
        super(Fuse_three,self).__init__()
        self.mask_2 = nn.Sequential(
                     nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, dilation=1,bias = True),
                     nn.Sigmoid()
                     )
        self.mask_3 = nn.Sequential(
                     nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, dilation=1,bias = True),
                     nn.Sigmoid()
                     )
    def forward(self,x1,x2,x3):
        mask_2 = self.mask_2(x2) # x1 is main flow
        mask_3 = self.mask_3(x3)
        return x1 * (2 - mask_2 - mask_3) + x2 * mask_2 + x3 * mask_3
    
def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    # Acknowledgment to FAIR's internal code
    kaiming_init(
        module,
        a=1,
        mode='fan_in',
        nonlinearity='leaky_relu',
        distribution='uniform')

def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

class Up2X(nn.Module): # size 2X, channel 1/2X
    def __init__(self,in_channel):
        super(Up2X,self).__init__()
        self.up = nn.Sequential(
                     nn.Conv2d(in_channel, in_channel//2, 3, stride=1, padding=1, dilation=1,bias = False),
                     nn.BatchNorm2d(in_channel//2),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
    def forward(self,x):
        x = self.up(x)
        return x

class Down2X(nn.Module): # size 1/2X, channel 2X
    def __init__(self,in_channel):
        super(Down2X,self).__init__()
        self.down = nn.Sequential(
                     nn.Conv2d(in_channel, in_channel*2, 3, stride=2, padding=1, dilation=1,bias = False),
                     nn.BatchNorm2d(in_channel*2),
                     nn.ReLU(inplace=True),
                     )
    def forward(self,x):
        x = self.down(x)
        return x


class CBR(nn.Module):
    '''
    input:[256,32,32],[128,64,64],[64,128,128]
    '''
    def __init__(self,in_channel,out_channel,dilation):
        super(CBR,self).__init__()
        self.cbr = nn.Sequential(
                    nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1, dilation=dilation, bias = False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True),
                    )
    def forward(self,x):
        x = self.cbr(x)
        return x

class ResUnit(nn.Module):
    def __init__(self,in_channel,out_channel,number,dilation=1):
        super(ResUnit,self).__init__()
        self.resunit = nn.ModuleList()
        for _ in range(number):
            self.resunit.append(CBR(in_channel,out_channel,dilation=dilation))
    def forward(self,x):
        res = x
        for m in self.resunit:
            x = m(x)
        return x + res

class Head_hm(nn.Module): # size 2X, channel 1/2X
    def __init__(self):
        super(Head_hm,self).__init__()
        self.layer1 = nn.Sequential(
                     nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1,bias = True),
                     #nn.BatchNorm2d(128),
                     nn.ReLU(inplace=True),
                     )
        self.layer2 = nn.Sequential(
                     nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1,bias = True),
                     #nn.BatchNorm2d(128),
                     nn.ReLU(inplace=True),
                     )
        self.layer3 = nn.Sequential(
                     nn.Conv2d(128, 80, 1, stride=1, padding=0, dilation=1,bias = True),
                     )
        self.init_weights()

    def init_weights(self):

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.layer3[0], std=0.01, bias=bias_cls)


    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Head_wh(nn.Module): # size 2X, channel 1/2X.
    def __init__(self):
        super(Head_wh,self).__init__()
        self.layer1 = nn.Sequential(
                     nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1,bias = True),
                     #nn.BatchNorm2d(64),
                     nn.ReLU(inplace=True),
                     )
        self.layer2 = nn.Sequential(
                     nn.Conv2d(128, 64, 3, stride=1, padding=1, dilation=1,bias = True),
                     #nn.BatchNorm2d(64),
                     nn.ReLU(inplace=True),
                     )
        self.layer3 = nn.Sequential(
                     nn.Conv2d(64, 4, 1, stride=1, padding=0, dilation=1,bias = True),
                     )
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



class DH(nn.Module):
    def __init__(self,backbone):
        super(DH, self).__init__()
        self.wh_offset_base = 16
        # heads
        self.hm = Head_hm()
        self.wh = Head_wh()

        # 1 for 512 channel
        self.mdcn1 = nn.Sequential(
                     DCNv2(512, 256, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     #nn.Conv2d(512, 256, 3, stride=1, padding=1, dilation=1),
                     nn.BatchNorm2d(256),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
        self.mdcn2 = nn.Sequential(
                     DCNv2(256, 128, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     #nn.Conv2d(256, 128, 3, stride=1, padding=1, dilation=1),
                     nn.BatchNorm2d(128),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
        self.init_weights()

    def init_weights(self):
        for _, m in self.mdcn1.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for _, m in self.mdcn2.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        x2 = x2 + self.mdcn1(x1)
        x3 = x3 + self.mdcn2(x2)

        hm = self.hm(x3)
        wh = F.relu(self.wh(x3)) * self.wh_offset_base
        return hm, wh


class DDH(nn.Module):
    def __init__(self,backbone,affm=False):
        super(DDH, self).__init__()
        self.wh_offset_base = 16
        self.affm = affm
        # heads
        self.hm = Head_hm()
        self.wh = Head_wh()

        if self.affm == True:
            # sum attention, heatmap and wh
            self.Fuse_two_hm1 = Fuse_two(in_channel = 256)
            self.Fuse_two_hm2 = Fuse_two(in_channel = 128)
            self.Fuse_two_wh1 = Fuse_two(in_channel = 256)
            self.Fuse_two_wh2 = Fuse_two(in_channel = 128)

        # 1 for 512 channel
        self.mdcn1 = nn.Sequential(
                     DCNv2(512, 256, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(256),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
        self.mdcn2 = nn.Sequential(
                     DCNv2(256, 128, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(128),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )


        self.mdcn1_hm = nn.Sequential(
                     DCNv2(512, 256, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(256),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
        self.mdcn2_hm = nn.Sequential(
                     DCNv2(256, 128, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(128),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )


        self.mdcn1_wh = nn.Sequential(
                     DCNv2(512, 256, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(256),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
        self.mdcn2_wh = nn.Sequential(
                     DCNv2(256, 128, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(128),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
        
        self.init_weights()

    def init_weights(self):

        for _, m in self.mdcn1.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for _, m in self.mdcn2.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for _, m in self.mdcn1_hm.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for _, m in self.mdcn2_hm.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for _, m in self.mdcn1_wh.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for _, m in self.mdcn2_wh.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        x2 = x2 + self.mdcn1(x1)
        x3 = x3 + self.mdcn2(x2)

        if self.affm == True:
            x_hm = self.mdcn1_hm(x1)
            x_hm = self.mdcn2_hm(self.Fuse_two_hm1(x_hm,x2)) # attention is computed on x2
            x_hm = self.Fuse_two_hm2(x_hm,x3)

            x_wh = self.mdcn1_wh(x1)
            x_wh = self.mdcn2_wh(self.Fuse_two_wh1(x_wh,x2))
            x_wh = self.Fuse_two_wh2(x_wh,x3)
        else:
            x_hm = self.mdcn1_hm(x1)
            x_hm = self.mdcn2_hm(x_hm + x2) # attention is computed on x2
            x_hm = x_hm + x3

            x_wh = self.mdcn1_wh(x1)
            x_wh = self.mdcn2_wh(x_wh + x2)
            x_wh = x_wh + x3

        hm = self.hm(x_hm)
        wh = F.relu(self.wh(x_wh)) * self.wh_offset_base
        return hm, wh
    
class LBFFM(nn.Module):

    def __init__(self, backbone, affm = False):
        super(LBFFM, self).__init__()
        self.backbone = backbone
        self.affm = affm

        self.down21_1 = Down2X(in_channel = 256)
        self.down32_1 = Down2X(in_channel = 128)
        self.up12_1 = Up2X(in_channel = 512)
        self.up23_1 = Up2X(in_channel = 256)

        self.down21_2 = Down2X(in_channel = 256)
        self.down32_2 = Down2X(in_channel = 128)
        self.up12_2 = Up2X(in_channel = 512)
        self.up23_2 = Up2X(in_channel = 256)

        self.down21_3 = Down2X(in_channel = 256)
        self.down32_3 = Down2X(in_channel = 128)
        self.up12_3 = Up2X(in_channel = 512)
        self.up23_3 = Up2X(in_channel = 256)

        if self.affm == True:
            # fuse attention
            self.Fuse1_1 = Fuse_two(in_channel = 512)
            self.Fuse1_2 = Fuse_three(in_channel = 256)
            self.Fuse1_3 = Fuse_two(in_channel = 128)

            self.Fuse2_1 = Fuse_two(in_channel = 512)
            self.Fuse2_2 = Fuse_three(in_channel = 256)
            self.Fuse2_3 = Fuse_two(in_channel = 128)

            self.Fuse3_1 = Fuse_two(in_channel = 512)
            self.Fuse3_2 = Fuse_three(in_channel = 256)
            self.Fuse3_3 = Fuse_two(in_channel = 128)

        if self.backbone == "resnet50" or self.backbone == "resnet101" or self.backbone == "resnet152" or self.backbone == "res2net101":
            self.shortcut1_0 = nn.Sequential(
                                CBR(in_channel = 2048,out_channel = 512,dilation = 1),
                                )
            self.shortcut2_0 = nn.Sequential(
                                CBR(in_channel = 1024,out_channel = 256,dilation = 1),
                                )
            self.shortcut3_0 = nn.Sequential(
                                CBR(in_channel = 512,out_channel = 128,dilation = 1),
                                )

        if self.backbone == "darknet53":
            self.shortcut1_0 = nn.Sequential(
                                CBR(in_channel = 1024,out_channel = 512,dilation = 1),
                                )
            self.shortcut2_0 = nn.Sequential(
                                CBR(in_channel = 512,out_channel = 256,dilation = 1),
                                )
            self.shortcut3_0 = nn.Sequential(
                                CBR(in_channel = 256,out_channel = 128,dilation = 1),  
                                )


        self.shortcut1_1 = nn.Sequential(
                            ResUnit(in_channel = 512,out_channel = 512,number = 2,dilation = 1),
                            CBR(in_channel = 512,out_channel = 512,dilation = 1),
                            )
        self.shortcut2_1 = nn.Sequential(
                            ResUnit(in_channel = 256,out_channel = 256,number = 2,dilation = 1),
                            CBR(in_channel = 256,out_channel = 256,dilation = 1),
                            )


        self.shortcut1_2 = nn.Sequential(
                            ResUnit(in_channel = 512,out_channel = 512,number = 2,dilation = 1),
                            CBR(in_channel = 512,out_channel = 512,dilation = 1),
                            )
        self.shortcut3_2 = nn.Sequential(
                            ResUnit(in_channel = 128,out_channel = 128,number = 2,dilation = 1),
                            CBR(in_channel = 128,out_channel = 128,dilation = 1),
                            )


        self.shortcut2_3 = nn.Sequential(
                            ResUnit(in_channel = 256,out_channel = 256,number = 2,dilation = 1),
                            CBR(in_channel = 256,out_channel = 256,dilation = 1),
                            )
        self.shortcut3_3 = nn.Sequential(
                            ResUnit(in_channel = 128,out_channel = 128,number = 2,dilation = 1),
                            CBR(in_channel = 128,out_channel = 128,dilation = 1),
                            )

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x1 = feats[-1] # 512 channel
        x2 = feats[-2] # 256 channel
        x3 = feats[-3] # 128 channel
        #x4 = feats[-4] # 128 channel

        if self.backbone == "resnet50" or self.backbone == "resnet101" or self.backbone == "resnet152" or self.backbone == "res2net101"\
            or self.backbone == "darknet53":
            x1 = self.shortcut1_0(x1)
            x2 = self.shortcut2_0(x2)
            x3 = self.shortcut3_0(x3)

        if self.affm == True:
            x1 = self.shortcut1_1(self.Fuse1_1(x1, self.down21_1(x2))) # attention used on up and down, not main flow
            x2 = self.shortcut2_1(self.Fuse1_2(x2, self.up12_1(x1), self.down32_1(x3)))
            x3 = self.Fuse1_3(x3, self.up23_1(x2))

            x1 = self.shortcut1_2(self.Fuse2_1(x1, self.down21_2(x2)))
            x2 = self.Fuse2_2(x2, self.up12_2(x1), self.down32_2(x3))
            x3 = self.shortcut3_2(self.Fuse2_3(x3, self.up23_2(x2)))

            x1 = self.Fuse3_1(x1, self.down21_3(x2))
            x2 = self.shortcut2_3(self.Fuse3_2(x2, self.up12_3(x1), self.down32_3(x3)))
            x3 = self.shortcut3_3(self.Fuse3_3(x3, self.up23_3(x2)))

        else:
            x1 = self.shortcut1_1(x1 + self.down21_1(x2)) # attention used on up and down, not main flow
            x2 = self.shortcut2_1(x2 + self.up12_1(x1) + self.down32_1(x3))
            x3 = x3 + self.up23_1(x2)

            x1 = self.shortcut1_2(x1 + self.down21_2(x2))
            x2 = x2 + self.up12_2(x1) + self.down32_2(x3)
            x3 = self.shortcut3_2(x3 + self.up23_2(x2))

            x1 = x1 + self.down21_3(x2)
            x2 = self.shortcut2_3(x2 + self.up12_3(x1) + self.down32_3(x3))
            x3 = self.shortcut3_3(x3 + self.up23_3(x2))

        return x1,x2,x3

class Stronger_CenterNet(nn.Module):
    def __init__(self,backbone='resnet18',affm = False,ddh = False):
        assert backbone == 'resnet18' or backbone == 'resnet34' or backbone == 'resnet50' or backbone == 'resnet101' or backbone == 'resnet152' or backbone == 'res2net101'\
               or backbone == 'darknet53'
        super(Stronger_CenterNet, self).__init__()
        self.affm = affm
        self.ddh = ddh
        if backbone == "resnet18":
            self.backbone = resnet18(pretrained=True)
        if backbone == "resnet34":
            self.backbone = resnet34(pretrained=True)
        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=True)
        if backbone == "resnet101":
            self.backbone = resnet101(pretrained=True)
        if backbone == "resnet152":
            self.backbone = resnet152(pretrained=True)
        if backbone == "res2net101":
            self.backbone = Res2Net(depth=101,frozen_stages = 1)
            self.backbone.init_weights(pretrained = "./pretrain/res2net101_v1b_26w_4s-0812c246.pth")

        if backbone == "darknet53":
            self.backbone = darknet53(pretrained = True)

        self.neck = LBFFM(backbone = backbone, affm = self.affm)
        if self.ddh == True:
            self.head = DDH(backbone = backbone,affm = self.affm)
        else:
            self.head = DH(backbone = backbone)

    def forward(self,x):

        return self.head(self.neck(self.backbone(x)))


if __name__ == '__main__':

    import torch
    import cv2
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    from thop import profile,clever_format

    inp = cv2.imread('./test.jpg')
    inp = cv2.resize(inp,(768,768))
    inp = cv2.cvtColor(inp,cv2.COLOR_BGR2RGB)
    inp = (inp.astype(np.float32) / 255.)
    mean = np.array([0.485, 0.456, 0.406],dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(1, 1, 3)
    inp = (inp - mean) / std 
    inp = inp.transpose(2, 0, 1)

    inp = torch.from_numpy(inp)
    inp = inp.unsqueeze(dim=0).cuda()
    #print(inp.shape)
    model = Stronger_CenterNet(backbone='resnet152',affm = True,ddh = True).cuda()
    #print(model)
    hm,wh = model(inp)
    print(hm.size())
    print(wh.size())
    #print(hm)

    macs,params = profile(model,inputs=(inp,))
    macs,params = clever_format([macs,params],"%.3f")
    print(macs,params)


