import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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

class ClssNet_svm(nn.Module):

    def __init__(self, n_class=21, nodeconv=False):
        super(ClssNet_svm, self).__init__()
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

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Linear(1024, n_class),
        )

        self.final_fc = nn.Sequential(
            nn.Linear(1024, n_class),
        )


        self.another_fc = nn.Sequential(
            nn.Linear(8*8*512,1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        # self.final_fc = nn.Sequential(
        #     nn.Linear(1024,n_class),
        # )

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

        # hh  = self.classifier_fc(hc5_f)
        feature  = self.another_fc(hc5_f)
        hh = self.final_fc(feature)


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
        return F.log_softmax(hh),feature

    def copy_params_from_self(self):

        self.another_fc[0].weight.data = self.classifier_fc[0].weight.data
        self.another_fc[0].bias.data = self.classifier_fc[0].bias.data
        #
        self.another_fc[3].weight.data = self.classifier_fc[3].weight.data
        self.another_fc[3].bias.data = self.classifier_fc[3].bias.data
        #

        self.final_fc[0].weight.data = self.classifier_fc[6].weight.data
        self.final_fc[0].bias.data = self.classifier_fc[6].bias.data



    def copy_params_from_old(self, oldmodel, copy_classifier=True):

        self.conv1[0].weight.data   = oldmodel.conv1[0].weight.data;
        self.conv1[0].bias.data     = oldmodel.conv1[0].bias.data;
        self.conv1[2].weight.data   = oldmodel.conv1[2].weight.data;
        self.conv1[2].bias.data     = oldmodel.conv1[2].bias.data;


        self.conv2[0].weight.data   = oldmodel.conv2[0].weight.data;
        self.conv2[0].bias.data     = oldmodel.conv2[0].bias.data;
        self.conv2[2].weight.data   = oldmodel.conv2[2].weight.data;
        self.conv2[2].bias.data     = oldmodel.conv2[2].bias.data;


        self.conv3[0].weight.data   = oldmodel.conv3[0].weight.data;
        self.conv3[0].bias.data     = oldmodel.conv3[0].bias.data;
        self.conv3[2].weight.data   = oldmodel.conv3[2].weight.data;
        self.conv3[2].bias.data     = oldmodel.conv3[2].bias.data;
        self.conv3[4].weight.data   = oldmodel.conv3[4].weight.data;
        self.conv3[4].bias.data     = oldmodel.conv3[4].bias.data;


        self.conv4[0].weight.data   = oldmodel.conv4[0].weight.data;
        self.conv4[0].bias.data     = oldmodel.conv4[0].bias.data;
        self.conv4[2].weight.data   = oldmodel.conv4[2].weight.data;
        self.conv4[2].bias.data     = oldmodel.conv4[2].bias.data;
        self.conv4[4].weight.data   = oldmodel.conv4[4].weight.data;
        self.conv4[4].bias.data     = oldmodel.conv4[4].bias.data;

        self.conv5[0].weight.data   = oldmodel.conv5[0].weight.data;
        self.conv5[0].bias.data     = oldmodel.conv5[0].bias.data;
        self.conv5[2].weight.data   = oldmodel.conv5[2].weight.data;
        self.conv5[2].bias.data     = oldmodel.conv5[2].bias.data;
        self.conv5[4].weight.data   = oldmodel.conv5[4].weight.data;
        self.conv5[4].bias.data     = oldmodel.conv5[4].bias.data;

        if copy_classifier:
            self.classifier_fc[0].weight.data = oldmodel.classifier_fc[0].weight.data
            self.classifier_fc[0].bias.data = oldmodel.classifier_fc[0].bias.data
            #
            self.classifier_fc[3].weight.data = oldmodel.classifier_fc[3].weight.data
            self.classifier_fc[3].bias.data = oldmodel.classifier_fc[3].bias.data
            #
            self.final_fc[0].weight.data = oldmodel.classifier_fc[6].weight.data
            self.final_fc[0].bias.data = oldmodel.classifier_fc[6].bias.data

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