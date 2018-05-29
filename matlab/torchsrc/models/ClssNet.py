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




class ClssNet(nn.Module):

    def __init__(self, n_class=21, nodeconv=False):
        super(ClssNet, self).__init__()
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
            nn.Conv2d(512, 1024, 7, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # fc7
            nn.Conv2d(1024, 1024, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # score_fr
            nn.Conv2d(1024, n_class, 1, padding=1),
        )

        self.maxPool_fc = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.BatchNorm2d(512)
        )

        self.classifier_fc = nn.Sequential(
            nn.Linear(8*8*512,1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Linear(1024,n_class),
        )

        self.upscore = nn.Sequential(
            # Decoder 
            # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
            #                          stride=1, padding=0, output_padding=0,
            #                          groups=1, bias=True)
            # output_height = (height-1)*stride + kernel_size - 2*padding + output_padding
            # batch x 512 -> batch x 1 x 28 x 28

            #_upscore_layer 4
            nn.ConvTranspose2d(n_class,n_class,4,stride=2,padding=1,output_padding=0,bias=False),
        )

        self.score4 = nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size,
            #                 stride=1, padding=0, dilation=1,
            #                 groups=1, bias=True)
            # batch x 1 x 28 x 28 -> batch x 512
            nn.Conv2d(512, n_class, 1, stride=1, padding=0),
        )


        self.score3 = nn.Sequential(
            nn.Conv2d(256, n_class, 1, stride=1, padding=0),
        )

        self.score2 = nn.Sequential(
            nn.Conv2d(128, n_class, 1, stride=1, padding=0),
        )

        self.score1 = nn.Sequential(
            nn.Conv2d(64, n_class, 1, stride=1, padding=0),
        )



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

        #print("input size = %s"%(str(x.size())))
        hc1 = self.conv1(x)
        #print("conv1 size = %s"%(str(hc1.size())))
        hc2 = self.conv2(hc1)
        #print("conv2 size = %s"%(str(hc2.size())))
        hc3 = self.conv3(hc2)
        #print("conv3 size = %s"%(str(hc3.size())))
        hc4 = self.conv4(hc3)
        #print("conv4 size = %s"%(str(hc4.size())))
        hc5 = self.conv5(hc4)
        #print("conv5 size = %s"%(str(hc5.size())))
        hc5_f = self.maxPool_fc(hc5)
        hc5_f = hc5_f.view(-1,8*8*512)

        hh  = self.classifier_fc(hc5_f)

        # ha  = self.classifier(hc5)
        # #print("classifer size = %s"%(str(ha.size())))

        # hs4 = self.score4(hc4)
        # hd4 = self.upscore(ha)
        # hf4 = torch.add(hs4, hd4)
        # #print("deconv4 size = %s"%(str(hf4.size())))

        # hs3 = self.score3(hc3)
        # hd3 = self.upscore(hf4)
        # hf3 = torch.add(hs3, hd3)
        # #print("deconv3 size = %s"%(str(hf3.size())))

        # hs2 = self.score2(hc2)
        # hd2 = self.upscore(hf3)
        # hf2 = torch.add(hs2, hd2)
        # #print("deconv2 size = %s"%(str(hf2.size())))

        # hs1 = self.score1(hc1)
        # hd1 = self.upscore(hf2)
        # hf1 = torch.add(hs1, hd1)
        # #print("deconv1 size = %s"%(str(hf1.size())))

        # h = self.upscore(hf1)
        # #print("output size = %s"%(str(h.size())))
        return F.log_softmax(hh)

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
