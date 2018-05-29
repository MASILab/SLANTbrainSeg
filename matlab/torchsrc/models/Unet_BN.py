import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://discuss.pytorch.org/t/unet-implementation/426
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size, padding=1),
            nn.BatchNorm2d(out_size),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size, padding=1),
            nn.BatchNorm2d(out_size),
        )
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))

        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, 2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(out_size),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size, padding=1),
            nn.BatchNorm2d(out_size),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size, padding=1),
            nn.BatchNorm2d(out_size),
        )
        self.activation = activation

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))

        return out


class Unet_BN(nn.Module):
    def __init__(self, n_class):
        super(Unet_BN, self).__init__()

        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = UNetConvBlock(3, 64) #yuankai change 1 to 3
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)
        self.conv_block512_1024 = UNetConvBlock(512, 1024)

        self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)

        self.last = nn.Conv2d(64, n_class, 1)


    def forward(self, x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)
        # print("pool1 size = %s"%(str(pool1.size())))

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)
        # print("pool2 size = %s"%(str(pool2.size())))

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)
        # print("pool3 size = %s"%(str(pool3.size())))

        block4 = self.conv_block256_512(pool3)
        pool4 = self.pool4(block4)
        # print("pool4 size = %s"%(str(pool4.size())))

        block5 = self.conv_block512_1024(pool4)
        # print("block5 size = %s"%(str(block5.size())))

        up1 = self.up_block1024_512(block5, block4)
        # print("up1 size = %s"%(str(up1.size())))

        up2 = self.up_block512_256(up1, block3)
        # print("up2 size = %s"%(str(up2.size())))

        up3 = self.up_block256_128(up2, block2)
        # print("up3 size = %s"%(str(up3.size())))

        up4 = self.up_block128_64(up3, block1)
        # print("up4 size = %s"%(str(up4.size())))

        # return F.log_softmax(self.last(up4))
        return self.last(up4)