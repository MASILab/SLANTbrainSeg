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


def _score_layer(bottom, name, num_classes):
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        # print name,bottom.get_shape().as_list()
        shape = [1, 1, in_features, num_classes]
        # He initialization Sheme
        if name == "score_fr":
            num_input = in_features
            stddev = (2 / num_input) ** 0.5
        # elif name == "score_pool4":
        #    stddev = 0.001
        # elif name == "score_pool3":
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


class FCNUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=1):
        super(FCNUpBlock, self).__init__()

        self.upscore = nn.ConvTranspose2d(out_size, out_size, 4, stride=2, padding=1, output_padding=0, bias=True)
        self.score = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=0)

    def forward(self, x, bridge):
        # print("x size = %s"%(str(x.size())))
        # print("bridge size = %s"%(str(bridge.size())))
        up_score = self.upscore(x)
        # print("up_score size = %s"%(str(up_score.size())))
        bridge_score = self.score(bridge)
        # print("bridge_score size = %s"%(str(bridge.size())))
        out = torch.add(bridge_score, up_score)
        return out


class MTL_BN(nn.Module):
    def __init__(self, n_class=11, n_lmk=4, n_networks=5, nodeconv=False):
        super(MTL_BN, self).__init__()
        self.nodeconv = nodeconv
        self.n_networks = n_networks
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
        )

        self.conv2 = nn.Sequential(
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
        )

        self.conv3 = nn.Sequential(
            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
        )

        self.conv4 = nn.Sequential(
            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
        )

        self.conv5 = nn.Sequential(
            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/32
        )

        # lmk detection
        self.classifier_fc = nn.Sequential(
            nn.Linear(8 * 8 * 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Linear(512, n_class),
        )

        self.maxPool_fc = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.BatchNorm2d(512)
        )

        self.output_final_1 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1, output_padding=0, bias=True)
        self.output_final_2 = nn.ConvTranspose2d(4, 4, 4, stride=2, padding=1, output_padding=0, bias=True)
        self.output_final_3 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1, output_padding=0, bias=True)
        self.output_final_4 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1, output_padding=0, bias=True)
        self.output_final_5 = nn.ConvTranspose2d(4, 4, 4, stride=2, padding=1, output_padding=0, bias=True)

        # classification
        self.classifier_conv = nn.Sequential(
            nn.Conv2d(512, 1024, 7, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            # fc7
            nn.Conv2d(1024, 1024, 1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.classifier_1 = nn.Sequential(
            nn.Conv2d(1024, 2, 1, padding=1),
            nn.BatchNorm2d(2)
        )

        self.classifier_2 = nn.Sequential(
            nn.Conv2d(1024, 4, 1, padding=1),
            nn.BatchNorm2d(4)
        )

        self.classifier_3 = nn.Sequential(
            nn.Conv2d(1024, 2, 1, padding=1),
            nn.BatchNorm2d(2)
        )

        self.classifier_4 = nn.Sequential(
            nn.Conv2d(1024, 2, 1, padding=1),
            nn.BatchNorm2d(2)
        )

        self.classifier_5 = nn.Sequential(
            nn.Conv2d(1024, 4, 1, padding=1),
            nn.BatchNorm2d(4)
        )

        self.up_block_512_1 = FCNUpBlock(512, 2)
        self.up_block_256_1 = FCNUpBlock(256, 2)
        self.up_block_128_1 = FCNUpBlock(128, 2)
        self.up_block_64_1 = FCNUpBlock(64, 2)

        self.up_block_512_2 = FCNUpBlock(512, 4)
        self.up_block_256_2 = FCNUpBlock(256, 4)
        self.up_block_128_2 = FCNUpBlock(128, 4)
        self.up_block_64_2 = FCNUpBlock(64, 4)

        self.up_block_512_3 = FCNUpBlock(512, 2)
        self.up_block_256_3 = FCNUpBlock(256, 2)
        self.up_block_128_3 = FCNUpBlock(128, 2)
        self.up_block_64_3 = FCNUpBlock(64, 2)

        self.up_block_512_4 = FCNUpBlock(512, 2)
        self.up_block_256_4 = FCNUpBlock(256, 2)
        self.up_block_128_4 = FCNUpBlock(128, 2)
        self.up_block_64_4 = FCNUpBlock(64, 2)

        self.up_block_512_5 = FCNUpBlock(512, 4)
        self.up_block_256_5 = FCNUpBlock(256, 4)
        self.up_block_128_5 = FCNUpBlock(128, 4)
        self.up_block_64_5 = FCNUpBlock(64, 4)

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

    def forward(self, x, method):

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



        if method == 'clss':
            hc5_f = self.maxPool_fc(hc5)
            hc5_f = hc5_f.view(-1, 8 * 8 * 512)
            clss = self.classifier_fc(hc5_f)
            pred_clss = F.log_softmax(clss)
            # print("clss")
            return pred_clss
        else:
            hc_fc = self.classifier_conv(hc5)
            if method == 'KidneyLong':
                # ['KidneyLong','KidneyTrans','LiverLong','SpleenLong','SpleenTrans']
                ha_1 = self.classifier_1(hc_fc)
                up_1_1 = self.up_block_512_1(ha_1, hc4)
                up_1_2 = self.up_block_256_1(up_1_1, hc3)
                up_1_3 = self.up_block_128_1(up_1_2, hc2)
                up_1_4 = self.up_block_64_1(up_1_3, hc1)
                pred_1_lmk = self.output_final_1(up_1_4)
                # print("KidneyLong")
                return pred_1_lmk

            elif method == 'KidneyTrans':
                ha_2 = self.classifier_2(hc_fc)
                up_2_1 = self.up_block_512_2(ha_2, hc4)
                up_2_2 = self.up_block_256_2(up_2_1, hc3)
                up_2_3 = self.up_block_128_2(up_2_2, hc2)
                up_2_4 = self.up_block_64_2(up_2_3, hc1)
                pred_2_lmk = self.output_final_2(up_2_4)
                # print("KidneyTrans")
                return pred_2_lmk

            elif method == 'LiverLong':
                ha_3 = self.classifier_3(hc_fc)
                up_3_1 = self.up_block_512_3(ha_3, hc4)
                up_3_2 = self.up_block_256_3(up_3_1, hc3)
                up_3_3 = self.up_block_128_3(up_3_2, hc2)
                up_3_4 = self.up_block_64_3(up_3_3, hc1)
                pred_3_lmk = self.output_final_3(up_3_4)
                # print("LiverLong")
                return pred_3_lmk

            elif method == 'SpleenLong':
                ha_4 = self.classifier_4(hc_fc)
                up_4_1 = self.up_block_512_4(ha_4, hc4)
                up_4_2 = self.up_block_256_4(up_4_1, hc3)
                up_4_3 = self.up_block_128_4(up_4_2, hc2)
                up_4_4 = self.up_block_64_4(up_4_3, hc1)
                pred_4_lmk = self.output_final_4(up_4_4)
                # print("SpleenLong")
                return pred_4_lmk

            elif method == 'SpleenTrans':
                ha_5 = self.classifier_5(hc_fc)
                up_5_1 = self.up_block_512_5(ha_5, hc4)
                up_5_2 = self.up_block_256_5(up_5_1, hc3)
                up_5_3 = self.up_block_128_5(up_5_2, hc2)
                up_5_4 = self.up_block_64_5(up_5_3, hc1)
                pred_5_lmk = self.output_final_5(up_5_4)
                # print("SpleenLong")
                return pred_5_lmk

    def copy_params_from_vgg16(self, vgg16, copy_classifier=True, copy_fc8=True, init_upscore=True):

        self.conv1[0].weight.data = vgg16.features[0].weight.data;
        self.conv1[0].bias.data = vgg16.features[0].bias.data;
        self.conv1[3].weight.data = vgg16.features[2].weight.data;
        self.conv1[3].bias.data = vgg16.features[2].bias.data;

        self.conv2[0].weight.data = vgg16.features[5].weight.data;
        self.conv2[0].bias.data = vgg16.features[5].bias.data;
        self.conv2[3].weight.data = vgg16.features[7].weight.data;
        self.conv2[3].bias.data = vgg16.features[7].bias.data;

        self.conv3[0].weight.data = vgg16.features[10].weight.data;
        self.conv3[0].bias.data = vgg16.features[10].bias.data;
        self.conv3[3].weight.data = vgg16.features[12].weight.data;
        self.conv3[3].bias.data = vgg16.features[12].bias.data;
        self.conv3[6].weight.data = vgg16.features[14].weight.data;
        self.conv3[6].bias.data = vgg16.features[14].bias.data;

        self.conv4[0].weight.data = vgg16.features[17].weight.data;
        self.conv4[0].bias.data = vgg16.features[17].bias.data;
        self.conv4[3].weight.data = vgg16.features[19].weight.data;
        self.conv4[3].bias.data = vgg16.features[19].bias.data;
        self.conv4[6].weight.data = vgg16.features[21].weight.data;
        self.conv4[6].bias.data = vgg16.features[21].bias.data;

        self.conv5[0].weight.data = vgg16.features[24].weight.data;
        self.conv5[0].bias.data = vgg16.features[24].bias.data;
        self.conv5[3].weight.data = vgg16.features[26].weight.data;
        self.conv5[3].bias.data = vgg16.features[26].bias.data;
        self.conv5[6].weight.data = vgg16.features[28].weight.data;
        self.conv5[6].bias.data = vgg16.features[28].bias.data;

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
