import numpy as np
import torch
import torch.nn as nn

#This is based on Zhoubing's simple net


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class DeconvNet(nn.Module):
    def __init__(self, n_class=21):
        super(DeconvNet, self).__init__()
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.pool6 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.unpool6 = nn.MaxUnpool2d(2, stride=2)


        self.conv_block1_32 = ConvBlock(3, 32) #yuankai change 1 to 3
        self.conv_block32_64 = ConvBlock(32, 64)
        self.conv_block64_128 = ConvBlock(64, 128)
        self.conv_block128_256 = ConvBlock(128, 256)
        self.conv_block256_512 = ConvBlock(256, 512)
        self.conv_block512_512 = ConvBlock(512, 512)

        self.deconv_block512_512 = ConvBlock(512, 512)
        self.deconv_block512_256 = ConvBlock(512, 256)
        self.deconv_block256_128 = ConvBlock(256, 128)
        self.deconv_block128_64 = ConvBlock(128, 64)
        self.deconv_block64_32 = ConvBlock(64, 32)
        self.deconv_block32_16 = ConvBlock(32, 16)

        self.last = nn.Conv2d(16, n_class, kernel_size=3, padding=1)




    def forward(self, x):
        # print("input size = %s"%(str(x.size())))
        x = self.conv_block1_32(x)
        x, indices1 = self.pool1(x)
        # print("conv1 size = %s" % (str(x.size())))
        x = self.conv_block32_64(x)
        x, indices2 = self.pool2(x)
        # print("conv2 size = %s" % (str(x.size())))
        x = self.conv_block64_128(x)
        x, indices3 = self.pool3(x)
        # print("conv3 size = %s" % (str(x.size())))
        x = self.conv_block128_256(x)
        x, indices4 = self.pool4(x)
        # print("conv4 size = %s" % (str(x.size())))
        x = self.conv_block256_512(x)
        x, indices5 = self.pool5(x)
        # print("conv5 size = %s" % (str(x.size())))
        x = self.conv_block512_512(x)
        x, indices6 = self.pool6(x)
        # print("conv6 size = %s" % (str(x.size())))

        x = self.unpool6(x, indices6)
        x = self.deconv_block512_512(x)
        # print("deconv1 size = %s" % (str(x.size())))
        x = self.unpool5(x, indices5)
        x = self.deconv_block512_256(x)
        # print("deconv2 size = %s" % (str(x.size())))
        x = self.unpool4(x, indices4)
        x = self.deconv_block256_128(x)
        # print("deconv3 size = %s" % (str(x.size())))
        x = self.unpool3(x, indices3)
        x = self.deconv_block128_64(x)
        # print("deconv4 size = %s" % (str(x.size())))
        x = self.unpool2(x, indices2)
        x = self.deconv_block64_32(x)
        # print("deconv5 size = %s" % (str(x.size())))
        x = self.unpool1(x, indices1)
        x = self.deconv_block32_16(x)
        # print("deconv6 size = %s" % (str(x.size())))
        x = self.last(x)

        return x