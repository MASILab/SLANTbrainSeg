import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


def _score_layer( bottom, name, num_classes):
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        #print name,bottom.get_shape().as_list()
        shape = [1, 1, in_features, num_classes]
        # He initialization Sheme
        if name == "score_fr":
            num_input = in_features
            stddev = (2 / num_input)**0.5
        #elif name == "score_pool4":
        #    stddev = 0.001
        #elif name == "score_pool3":
        #    stddev = 0.0001
        else:
            stddev = 0.001
        # Apply convolution
        w_decay = wd

        weights = _variable_with_weight_decay(shape, stddev, w_decay,
                                                   decoder=True)
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
        # Apply bias
        conv_biases = _bias_variable([num_classes], constant=0.0)
        bias = tf.nn.bias_add(conv, conv_biases)

        _activation_summary(bias)


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, 2, stride=2),
            nn.BatchNorm2d(out_size)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size, padding=1),
            nn.BatchNorm2d(out_size)
        )
        self.activation = activation

    def forward(self, x, bridge):
        # print("x = %s"%(str(x.size())))
        up = self.up(x)
        # print("up = %s"%(str(up.size())))
        # print("bridge = %s"%(str(bridge.size())))
        out = torch.cat([up, bridge], 1)
        out = self.activation(self.conv_1(out))
        return out

class Unet(nn.Module):

    def __init__(self, n_class=21, nodeconv=False):
        super(Unet, self).__init__()
        self.nodeconv = nodeconv
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
        )

        self.conv2 = nn.Sequential(
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
        )


        self.conv3 = nn.Sequential(
            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
        )
        
        self.conv4 = nn.Sequential(     
            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
        )

        self.conv5 = nn.Sequential(
            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/32
        )

        self.classifier = nn.Sequential(
            # fc6
            nn.Conv2d(512, 4096, 7, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # fc7
            nn.Conv2d(4096, 4096, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # score_fr
            nn.Conv2d(4096, n_class, 1, padding=1),
        )

        self.maxPool_fc = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier_fc = nn.Sequential(
            nn.Linear(8*8*512,4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Linear(4096,n_class),
        )

        self.conv_block512_1024 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024)
        )

        self.conv_block512_numclass = nn.Sequential(
            nn.Conv2d(512, n_class, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_class)
        )

        self.up_blocknumclass_512 = UNetUpBlock(n_class,n_class)
        self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)
        self.up_sample_256_512 =  nn.ConvTranspose2d(64, n_class, 2, stride=2)


        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        # print("input size = %s"%(str(x.size())))
        hc1 = self.conv1(x)
        # print("conv1 size = %s"%(str(hc1.size())))
        hc2 = self.conv2(hc1)
        # print("conv2 size = %s"%(str(hc2.size())))
        hc3 = self.conv3(hc2)
        # print("conv3 size = %s"%(str(hc3.size())))
        hc4 = self.conv4(hc3)
        # print("conv4 size = %s"%(str(hc4.size())))
        hc5 = self.conv5(hc4)
        # print("conv5 size = %s"%(str(hc5.size())))



        lowest = self.conv_block512_1024(hc5)
        #lowest = self.conv_block512_numclass(hc5)
        #print("lowest size = %s"%(str(lowest.size())))
        up1 = self.up_block1024_512(lowest, hc4)
        #up1 = self.up_blocknumclass_512(lowest, hc4)
        # print("up1 size = %s"%(str(up1.size())))
        up2 = self.up_block512_256(up1, hc3)
        # print("up2 size = %s"%(str(up2.size())))
        up3 = self.up_block256_128(up2, hc2)
        # print("up3 size = %s"%(str(up3.size())))
        up4 = self.up_block128_64(up3, hc1)
        # print("up4 size = %s"%(str(up4.size())))
        pred = self.up_sample_256_512(up4)
        # pred = self.last(up4)

        # print("pred size = %s"%(str(pred.size())))




        #I tensorflow/core/kernels/logging_ops.cc:79] Shape of input image: [8 512 512 3]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of pool1[8 256 256 64]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of pool2[8 128 128 128]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of pool3[8 64 64 256]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of pool4[8 32 32 512]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of pool5[8 16 16 512]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of fc6[8 16 16 4096]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of fc7[8 16 16 4096]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of fc8[8 16 16 3]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of upscore5[8 32 32 3]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of upscore4[8 64 64 3]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of upscore3[8 128 128 3]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of upscore2[8 256 256 3]
#I tensorflow/core/kernels/logging_ops.cc:79] Shape of upscore1[8 512 512 3]

        return pred

    def copy_params_from_vgg16(self, vgg16, copy_classifier=True, copy_fc8=True, init_upscore=True):

        self.conv1[0].weight.data   = vgg16.features[0].weight.data;
        self.conv1[0].bias.data     = vgg16.features[0].bias.data;
        # self.conv1[1].weight.data   = vgg16.features[1].weight.data;
        # self.conv1[1].bias.data     = vgg16.features[1].bias.data;
        self.conv1[2].weight.data   = vgg16.features[2].weight.data;
        self.conv1[2].bias.data     = vgg16.features[2].bias.data;
        # self.conv1[3].weight.data   = vgg16.features[3].weight.data;
        # self.conv1[3].bias.data     = vgg16.features[3].bias.data;
        # self.conv1[4].weight.data   = vgg16.features[4].weight.data;
        # self.conv1[4].bias.data     = vgg16.features[4].bias.data;

        self.conv2[0].weight.data   = vgg16.features[5].weight.data;
        self.conv2[0].bias.data     = vgg16.features[5].bias.data;
        # self.conv2[1].weight.data   = vgg16.features[6].weight.data;
        # self.conv2[1].bias.data     = vgg16.features[6].bias.data;
        self.conv2[2].weight.data   = vgg16.features[7].weight.data;
        self.conv2[2].bias.data     = vgg16.features[7].bias.data;
        # self.conv2[3].weight.data   = vgg16.features[8].weight.data;
        # self.conv2[3].bias.data     = vgg16.features[8].bias.data;
        # self.conv2[4].weight.data   = vgg16.features[9].weight.data;
        # self.conv2[4].bias.data     = vgg16.features[9].bias.data;

        self.conv3[0].weight.data   = vgg16.features[10].weight.data;
        self.conv3[0].bias.data     = vgg16.features[10].bias.data;
        # self.conv3[1].weight.data   = vgg16.features[11].weight.data;
        # self.conv3[1].bias.data     = vgg16.features[11].bias.data;
        self.conv3[2].weight.data   = vgg16.features[12].weight.data;
        self.conv3[2].bias.data     = vgg16.features[12].bias.data;
        # self.conv3[3].weight.data   = vgg16.features[13].weight.data;
        # self.conv3[3].bias.data     = vgg16.features[13].bias.data;
        self.conv3[4].weight.data   = vgg16.features[14].weight.data;
        self.conv3[4].bias.data     = vgg16.features[14].bias.data;
        # self.conv3[5].weight.data   = vgg16.features[15].weight.data;
        # self.conv3[5].bias.data     = vgg16.features[15].bias.data;
        # self.conv3[6].weight.data   = vgg16.features[16].weight.data;
        # self.conv3[6].bias.data     = vgg16.features[16].bias.data;

        self.conv4[0].weight.data   = vgg16.features[17].weight.data;
        self.conv4[0].bias.data     = vgg16.features[17].bias.data;
        # self.conv4[1].weight.data   = vgg16.features[18].weight.data;
        # self.conv4[1].bias.data     = vgg16.features[18].bias.data;
        self.conv4[2].weight.data   = vgg16.features[19].weight.data;
        self.conv4[2].bias.data     = vgg16.features[19].bias.data;
        # self.conv4[3].weight.data   = vgg16.features[20].weight.data;
        # self.conv4[3].bias.data     = vgg16.features[20].bias.data;
        self.conv4[4].weight.data   = vgg16.features[21].weight.data;
        self.conv4[4].bias.data     = vgg16.features[21].bias.data;
        # self.conv4[5].weight.data   = vgg16.features[22].weight.data;
        # self.conv4[5].bias.data     = vgg16.features[22].bias.data;
        # self.conv4[6].weight.data   = vgg16.features[23].weight.data;
        # self.conv4[6].bias.data     = vgg16.features[23].bias.data;

        self.conv5[0].weight.data   = vgg16.features[24].weight.data;
        self.conv5[0].bias.data     = vgg16.features[24].bias.data;
        # self.conv5[1].weight.data   = vgg16.features[25].weight.data;
        # self.conv5[1].bias.data     = vgg16.features[25].bias.data;
        self.conv5[2].weight.data   = vgg16.features[26].weight.data;
        self.conv5[2].bias.data     = vgg16.features[26].bias.data;
        # self.conv5[3].weight.data   = vgg16.features[27].weight.data;
        # self.conv5[3].bias.data     = vgg16.features[27].bias.data;
        self.conv5[4].weight.data   = vgg16.features[28].weight.data;
        self.conv5[4].bias.data     = vgg16.features[28].bias.data;
        # self.conv5[5].weight.data   = vgg16.features[29].weight.data;
        # self.conv5[5].bias.data     = vgg16.features[29].bias.data;
        # self.conv5[6].weight.data   = vgg16.features[30].weight.data;
        # self.conv5[6].bias.data     = vgg16.features[30].bias.data;
        if copy_classifier:
            for i in [0, 3]:
                l1 = vgg16.classifier[i]
                l2 = self.classifier[i]
                l2.weight.data = l1.weight.data.view(l2.weight.size())
                l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]
        if init_upscore:
            # initialize upscore layer
            c1, c2, h, w = self.upscore.weight.data.size()
            assert c1 == c2 == n_class
            assert h == w
            weight = get_upsample_filter(h)
            self.upscore.weight.data = \
                weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
