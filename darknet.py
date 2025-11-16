import torch
from torch import nn

# from https://github.com/developer0hye/PyTorch-Darknet53/blob/master/model.py

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU(negative_slope=0.1))


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes=1000):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        outs = []
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        outs.append(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        outs.append(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        outs.append(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        outs.append(out)
        # out = self.global_avg_pool(out)
        # out = out.view(-1, 1024)
        # out = self.fc(out)

        return outs

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet53(pretrained = True, num_classes=1000, model_weights = "./weights/backbone_weights/darknet53_76.28.pth"):
    model = Darknet53(DarkResidualBlock, num_classes)
    if pretrained == True:
        model.load_state_dict(torch.load(model_weights),strict = False)
        print("load weight from: {}".format(model_weights))
    return model

if __name__ == '__main__':
    model = darknet53()
    inp = torch.rand(1,3,256,256)
    out = model(inp)
    print(out[0].size(),out[1].size(),out[2].size(),out[3].size())