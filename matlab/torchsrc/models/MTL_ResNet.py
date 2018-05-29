import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch.nn.functional as F
#from pairwise import Pairwise

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        return out


class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out += shortcut
        out = self.relu(out)

        return out

class MTL_ResNet(nn.Module):
    def __init__(self, downblock, upblock, num_layers, n_classes, n_lmks):
        super(MTL_ResNet, self).__init__()

        print('######## of class:%d'%(n_classes))

        self.in_channels = 64
        self.in_channels = 64
        self.in_channels = 64
        self.in_channels = 64
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_downlayer(downblock, 64, num_layers[0])
        self.layer2 = self._make_downlayer(downblock, 128, num_layers[1],
                                            stride=2)
        self.layer3 = self._make_downlayer(downblock, 256, num_layers[2],
                                            stride=2)
        self.layer4 = self._make_downlayer(downblock, 512, num_layers[3],
                                            stride=2)

        in_channels_aftercovs = self.in_channels
        self.in_channels = in_channels_aftercovs
        self.uplayer1_1 = self._make_up_block(upblock, 512, 1, stride=2)
        self.uplayer2_1 = self._make_up_block(upblock, 256, num_layers[2], stride=2)
        self.uplayer3_1 = self._make_up_block(upblock, 128, num_layers[1], stride=2)
        self.uplayer4_1 = self._make_up_block(upblock, 64, 2, stride=2)
        upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256
                               64,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        )
        self.uplayer_top_1 = DeconvBottleneck(self.in_channels, 64, 1, 2, upsample_1)
        self.conv1_1_1 = nn.ConvTranspose2d(64, 2, kernel_size=1, stride=1, bias=False)

        self.in_channels = in_channels_aftercovs
        self.uplayer1_2 = self._make_up_block(upblock, 512, 1, stride=2)
        self.uplayer2_2 = self._make_up_block(upblock, 256, num_layers[2], stride=2)
        self.uplayer3_2 = self._make_up_block(upblock, 128, num_layers[1], stride=2)
        self.uplayer4_2 = self._make_up_block(upblock, 64, 2, stride=2)
        upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256
                               64,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        )
        self.uplayer_top_2 = DeconvBottleneck(self.in_channels, 64, 1, 2, upsample_2)
        self.conv1_1_2 = nn.ConvTranspose2d(64, 4, kernel_size=1, stride=1, bias=False)

        #
        self.in_channels = in_channels_aftercovs
        self.uplayer1_3 = self._make_up_block(upblock, 512, 1, stride=2)
        self.uplayer2_3 = self._make_up_block(upblock, 256, num_layers[2], stride=2)
        self.uplayer3_3 = self._make_up_block(upblock, 128, num_layers[1], stride=2)
        self.uplayer4_3 = self._make_up_block(upblock, 64, 2, stride=2)
        upsample_3 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256
                               64,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        )
        self.uplayer_top_3 = DeconvBottleneck(self.in_channels, 64, 1, 2, upsample_3)
        self.conv1_1_3 = nn.ConvTranspose2d(64, 2, kernel_size=1, stride=1, bias=False)
        #

        self.in_channels = in_channels_aftercovs
        self.uplayer1_4 = self._make_up_block(upblock, 512, 1, stride=2)
        self.uplayer2_4 = self._make_up_block(upblock, 256, num_layers[2], stride=2)
        self.uplayer3_4 = self._make_up_block(upblock, 128, num_layers[1], stride=2)
        self.uplayer4_4 = self._make_up_block(upblock, 64, 2, stride=2)
        upsample_4 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256
                               64,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        )
        self.uplayer_top_4 = DeconvBottleneck(self.in_channels, 64, 1, 2, upsample_4)
        self.conv1_1_4 = nn.ConvTranspose2d(64, 2, kernel_size=1, stride=1, bias=False)

        self.in_channels = in_channels_aftercovs
        self.uplayer1_5 = self._make_up_block(upblock, 512, 1, stride=2)
        self.uplayer2_5 = self._make_up_block(upblock, 256, num_layers[2], stride=2)
        self.uplayer3_5 = self._make_up_block(upblock, 128, num_layers[1], stride=2)
        self.uplayer4_5 = self._make_up_block(upblock, 64, 2, stride=2)
        upsample_5 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256
                               64,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        )
        self.uplayer_top_5 = DeconvBottleneck(self.in_channels, 64, 1, 2, upsample_5)
        self.conv1_1_5 = nn.ConvTranspose2d(64, 4, kernel_size=1, stride=1, bias=False)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * downblock.expansion * 4, n_classes)




 #       self.pw = Pairwise(neighbors=8, num_classes=n_classes)

    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels*block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels*2,
                                   kernel_size=1, stride=stride,
                                   bias=False, output_padding=1),
                nn.BatchNorm2d(init_channels*2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)

    def forward(self, x, method):
        img = x
        x_size = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if method == 'clss':
            clss = self.avgpool(x)
            clss = clss.view(clss.size(0), -1)
            clss = self.fc(clss)
            pred_clss = F.log_softmax(clss)
            return pred_clss

        elif method == 'all':
            lmk1 = self.uplayer1_1(x)
            lmk1 = self.uplayer2_1(lmk1)
            lmk1 = self.uplayer3_1(lmk1)
            lmk1 = self.uplayer4_1(lmk1)
            lmk1 = self.uplayer_top_1(lmk1)
            pred_1_lmk = self.conv1_1_1(lmk1, output_size=img.size())
            lmk2 = self.uplayer1_2(x)
            lmk2 = self.uplayer2_2(lmk2)
            lmk2 = self.uplayer3_2(lmk2)
            lmk2 = self.uplayer4_2(lmk2)
            lmk2 = self.uplayer_top_2(lmk2)
            pred_2_lmk = self.conv1_1_2(lmk2, output_size=img.size())
            lmk3 = self.uplayer1_3(x)
            lmk3 = self.uplayer2_3(lmk3)
            lmk3 = self.uplayer3_3(lmk3)
            lmk3 = self.uplayer4_3(lmk3)
            lmk3 = self.uplayer_top_3(lmk3)
            pred_3_lmk = self.conv1_1_3(lmk3, output_size=img.size())
            lmk4 = self.uplayer1_4(x)
            lmk4 = self.uplayer2_4(lmk4)
            lmk4 = self.uplayer3_4(lmk4)
            lmk4 = self.uplayer4_4(lmk4)
            lmk4 = self.uplayer_top_4(lmk4)
            pred_4_lmk = self.conv1_1_4(lmk4, output_size=img.size())
            lmk5 = self.uplayer1_5(x)
            lmk5 = self.uplayer2_5(lmk5)
            lmk5 = self.uplayer3_5(lmk5)
            lmk5 = self.uplayer4_5(lmk5)
            lmk5 = self.uplayer_top_5(lmk5)
            pred_5_lmk = self.conv1_1_5(lmk5, output_size=img.size())
            return pred_1_lmk, pred_2_lmk, pred_3_lmk, pred_4_lmk, pred_5_lmk

        else:
            if method == 'KidneyLong':
                # ['KidneyLong','KidneyTrans','LiverLong','SpleenLong','SpleenTrans']
                lmk1 = self.uplayer1_1(x)
                lmk1 = self.uplayer2_1(lmk1)
                lmk1 = self.uplayer3_1(lmk1)
                lmk1 = self.uplayer4_1(lmk1)
                lmk1 = self.uplayer_top_1(lmk1)
                pred_1_lmk = self.conv1_1_1(lmk1, output_size=img.size())
                return pred_1_lmk

            elif method == 'KidneyTrans':
                lmk2 = self.uplayer1_2(x)
                lmk2 = self.uplayer2_2(lmk2)
                lmk2 = self.uplayer3_2(lmk2)
                lmk2 = self.uplayer4_2(lmk2)
                lmk2 = self.uplayer_top_2(lmk2)
                pred_2_lmk = self.conv1_1_2(lmk2, output_size=img.size())
                return pred_2_lmk

            elif method == 'LiverLong':
                lmk3 = self.uplayer1_3(x)
                lmk3 = self.uplayer2_3(lmk3)
                lmk3 = self.uplayer3_3(lmk3)
                lmk3 = self.uplayer4_3(lmk3)
                lmk3 = self.uplayer_top_3(lmk3)
                pred_3_lmk = self.conv1_1_3(lmk3, output_size=img.size())
                return pred_3_lmk

            elif method == 'SpleenLong':
                lmk4 = self.uplayer1_4(x)
                lmk4 = self.uplayer2_4(lmk4)
                lmk4 = self.uplayer3_4(lmk4)
                lmk4 = self.uplayer4_4(lmk4)
                lmk4 = self.uplayer_top_4(lmk4)
                pred_4_lmk = self.conv1_1_4(lmk4, output_size=img.size())
                return pred_4_lmk

            elif method == 'SpleenTrans':
                lmk5 = self.uplayer1_5(x)
                lmk5 = self.uplayer2_5(lmk5)
                lmk5 = self.uplayer3_5(lmk5)
                lmk5 = self.uplayer4_5(lmk5)
                lmk5 = self.uplayer_top_5(lmk5)
                pred_5_lmk = self.conv1_1_5(lmk5, output_size=img.size())
                return pred_5_lmk


    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))


def MTL_ResNet50(pretrained=False,**kwargs):
    model = MTL_ResNet(Bottleneck, DeconvBottleneck, [3, 4, 6, 3],  **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        del state_dict['fc.weight'] #since we changed FC structure, remove this
        del state_dict['fc.bias']  # since we changed FC structure, remove this
        model.load_state_dict(state_dict)
    return model

def MTL_ResNet101(**kwargs):
    return MTL_ResNet(Bottleneck, DeconvBottleneck, [3, 4, 23, 2],  **kwargs)
