import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

import math


class GCN(nn.Module):
    def __init__(self, inplanes, planes, ks=7):
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, planes, kernel_size=(ks, 1),
                                 padding=(ks/2, 0))

        self.conv_l2 = nn.Conv2d(planes, planes, kernel_size=(1, ks),
                                 padding=(0, ks/2))
        self.conv_r1 = nn.Conv2d(inplanes, planes, kernel_size=(1, ks),
                                 padding=(0, ks/2))
        self.conv_r2 = nn.Conv2d(planes, planes, kernel_size=(ks, 1),
                                 padding=(ks/2, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x


class Refine(nn.Module):
    def __init__(self, planes):
        super(Refine, self).__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        out = residual + x
        return out


class MTL_GCN(nn.Module):
    def __init__(self, num_classes):
        super(MTL_GCN, self).__init__()

        self.num_classes = num_classes

        self.lmk2 = 2
        self.lmk4 = 4

        resnet = models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


        #Kidney Long
        self.gcn1_1 = GCN(2048, self.lmk2)
        self.gcn1_2 = GCN(1024, self.lmk2)
        self.gcn1_3 = GCN(512, self.lmk2)
        self.gcn1_4 = GCN(64, self.lmk2)
        self.gcn1_5 = GCN(64, self.lmk2)

        self.refine1_1 = Refine(self.lmk2)
        self.refine1_2 = Refine(self.lmk2)
        self.refine1_3 = Refine(self.lmk2)
        self.refine1_4 = Refine(self.lmk2)
        self.refine1_5 = Refine(self.lmk2)
        self.refine1_6 = Refine(self.lmk2)
        self.refine1_7 = Refine(self.lmk2)
        self.refine1_8 = Refine(self.lmk2)
        self.refine1_9 = Refine(self.lmk2)
        self.refine1_10 = Refine(self.lmk2)

        #Kidney Trans
        self.gcn2_1 = GCN(2048, self.lmk4)
        self.gcn2_2 = GCN(1024, self.lmk4)
        self.gcn2_3 = GCN(512, self.lmk4)
        self.gcn2_4 = GCN(64, self.lmk4)
        self.gcn2_5 = GCN(64, self.lmk4)

        self.refine2_1 = Refine(self.lmk4)
        self.refine2_2 = Refine(self.lmk4)
        self.refine2_3 = Refine(self.lmk4)
        self.refine2_4 = Refine(self.lmk4)
        self.refine2_5 = Refine(self.lmk4)
        self.refine2_6 = Refine(self.lmk4)
        self.refine2_7 = Refine(self.lmk4)
        self.refine2_8 = Refine(self.lmk4)
        self.refine2_9 = Refine(self.lmk4)
        self.refine2_10 = Refine(self.lmk4)

        #Liver Long
        self.gcn3_1 = GCN(2048, self.lmk2)
        self.gcn3_2 = GCN(1024, self.lmk2)
        self.gcn3_3 = GCN(512, self.lmk2)
        self.gcn3_4 = GCN(64, self.lmk2)
        self.gcn3_5 = GCN(64, self.lmk2)

        self.refine3_1 = Refine(self.lmk2)
        self.refine3_2 = Refine(self.lmk2)
        self.refine3_3 = Refine(self.lmk2)
        self.refine3_4 = Refine(self.lmk2)
        self.refine3_5 = Refine(self.lmk2)
        self.refine3_6 = Refine(self.lmk2)
        self.refine3_7 = Refine(self.lmk2)
        self.refine3_8 = Refine(self.lmk2)
        self.refine3_9 = Refine(self.lmk2)
        self.refine3_10 = Refine(self.lmk2)

        #Spleen Long
        self.gcn4_1 = GCN(2048, self.lmk2)
        self.gcn4_2 = GCN(1024, self.lmk2)
        self.gcn4_3 = GCN(512, self.lmk2)
        self.gcn4_4 = GCN(64, self.lmk2)
        self.gcn4_5 = GCN(64, self.lmk2)

        self.refine4_1 = Refine(self.lmk2)
        self.refine4_2 = Refine(self.lmk2)
        self.refine4_3 = Refine(self.lmk2)
        self.refine4_4 = Refine(self.lmk2)
        self.refine4_5 = Refine(self.lmk2)
        self.refine4_6 = Refine(self.lmk2)
        self.refine4_7 = Refine(self.lmk2)
        self.refine4_8 = Refine(self.lmk2)
        self.refine4_9 = Refine(self.lmk2)
        self.refine4_10 = Refine(self.lmk2)

        #Spleen Trans
        self.gcn5_1 = GCN(2048, self.lmk4)
        self.gcn5_2 = GCN(1024, self.lmk4)
        self.gcn5_3 = GCN(512, self.lmk4)
        self.gcn5_4 = GCN(64, self.lmk4)
        self.gcn5_5 = GCN(64, self.lmk4)

        self.refine5_1 = Refine(self.lmk4)
        self.refine5_2 = Refine(self.lmk4)
        self.refine5_3 = Refine(self.lmk4)
        self.refine5_4 = Refine(self.lmk4)
        self.refine5_5 = Refine(self.lmk4)
        self.refine5_6 = Refine(self.lmk4)
        self.refine5_7 = Refine(self.lmk4)
        self.refine5_8 = Refine(self.lmk4)
        self.refine5_9 = Refine(self.lmk4)
        self.refine5_10 = Refine(self.lmk4)

        # self.out0 = self._classifier(2048)
        # self.out1 = self._classifier(1024)
        # self.out2 = self._classifier(512)
        # self.out_e = self._classifier(256)
        # self.out3 = self._classifier(64)
        # self.out4 = self._classifier(64)
        # self.out5 = self._classifier(32)

        self.transformer = nn.Conv2d(256, 64, kernel_size=1)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * 4 * 4, self.num_classes)

    def _classifier(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes/2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes/2, self.num_classes, 1),
        )

    def forward(self, x, method):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        if method == 'clss':
            #classification
            clss = self.avgpool(fm4)
            clss = clss.view(clss.size(0), -1)
            clss = self.fc(clss)
            pred_clss = F.log_softmax(clss)
            return pred_clss

        elif method == 'all':
            gcfm1_1 = self.refine1_1(self.gcn1_1(fm4))
            gcfm1_2 = self.refine1_2(self.gcn1_2(fm3))
            gcfm1_3 = self.refine1_3(self.gcn1_3(fm2))
            gcfm1_4 = self.refine1_4(self.gcn1_4(pool_x))
            gcfm1_5 = self.refine1_5(self.gcn1_5(conv_x))

            fs1_1 = self.refine1_6(F.upsample_bilinear(gcfm1_1, fm3.size()[2:]) + gcfm1_2)
            fs1_2 = self.refine1_7(F.upsample_bilinear(fs1_1, fm2.size()[2:]) + gcfm1_3)
            fs1_3 = self.refine1_8(F.upsample_bilinear(fs1_2, pool_x.size()[2:]) + gcfm1_4)
            fs1_4 = self.refine1_9(F.upsample_bilinear(fs1_3, conv_x.size()[2:]) + gcfm1_5)
            pred_1_lmk = self.refine1_10(F.upsample_bilinear(fs1_4, input.size()[2:]))

            gcfm2_1 = self.refine2_1(self.gcn2_1(fm4))
            gcfm2_2 = self.refine2_2(self.gcn2_2(fm3))
            gcfm2_3 = self.refine2_3(self.gcn2_3(fm2))
            gcfm2_4 = self.refine2_4(self.gcn2_4(pool_x))
            gcfm2_5 = self.refine2_5(self.gcn2_5(conv_x))

            fs2_1 = self.refine2_6(F.upsample_bilinear(gcfm2_1, fm3.size()[2:]) + gcfm2_2)
            fs2_2 = self.refine2_7(F.upsample_bilinear(fs2_1, fm2.size()[2:]) + gcfm2_3)
            fs2_3 = self.refine2_8(F.upsample_bilinear(fs2_2, pool_x.size()[2:]) + gcfm2_4)
            fs2_4 = self.refine2_9(F.upsample_bilinear(fs2_3, conv_x.size()[2:]) + gcfm2_5)
            pred_2_lmk = self.refine2_10(F.upsample_bilinear(fs2_4, input.size()[2:]))

            gcfm3_1 = self.refine3_1(self.gcn3_1(fm4))
            gcfm3_2 = self.refine3_2(self.gcn3_2(fm3))
            gcfm3_3 = self.refine3_3(self.gcn3_3(fm2))
            gcfm3_4 = self.refine3_4(self.gcn3_4(pool_x))
            gcfm3_5 = self.refine3_5(self.gcn3_5(conv_x))

            fs3_1 = self.refine3_6(F.upsample_bilinear(gcfm3_1, fm3.size()[2:]) + gcfm3_2)
            fs3_2 = self.refine3_7(F.upsample_bilinear(fs3_1, fm2.size()[2:]) + gcfm3_3)
            fs3_3 = self.refine3_8(F.upsample_bilinear(fs3_2, pool_x.size()[2:]) + gcfm3_4)
            fs3_4 = self.refine3_9(F.upsample_bilinear(fs3_3, conv_x.size()[2:]) + gcfm3_5)
            pred_3_lmk = self.refine3_10(F.upsample_bilinear(fs3_4, input.size()[2:]))

            gcfm4_1 = self.refine4_1(self.gcn4_1(fm4))
            gcfm4_2 = self.refine4_2(self.gcn4_2(fm3))
            gcfm4_3 = self.refine4_3(self.gcn4_3(fm2))
            gcfm4_4 = self.refine4_4(self.gcn4_4(pool_x))
            gcfm4_5 = self.refine4_5(self.gcn4_5(conv_x))

            fs4_1 = self.refine4_6(F.upsample_bilinear(gcfm4_1, fm3.size()[2:]) + gcfm4_2)
            fs4_2 = self.refine4_7(F.upsample_bilinear(fs4_1, fm2.size()[2:]) + gcfm4_3)
            fs4_3 = self.refine4_8(F.upsample_bilinear(fs4_2, pool_x.size()[2:]) + gcfm4_4)
            fs4_4 = self.refine4_9(F.upsample_bilinear(fs4_3, conv_x.size()[2:]) + gcfm4_5)
            pred_4_lmk = self.refine4_10(F.upsample_bilinear(fs4_4, input.size()[2:]))

            gcfm5_1 = self.refine5_1(self.gcn5_1(fm4))
            gcfm5_2 = self.refine5_2(self.gcn5_2(fm3))
            gcfm5_3 = self.refine5_3(self.gcn5_3(fm2))
            gcfm5_4 = self.refine5_4(self.gcn5_4(pool_x))
            gcfm5_5 = self.refine5_5(self.gcn5_5(conv_x))

            fs5_1 = self.refine5_6(F.upsample_bilinear(gcfm5_1, fm3.size()[2:]) + gcfm5_2)
            fs5_2 = self.refine5_7(F.upsample_bilinear(fs5_1, fm2.size()[2:]) + gcfm5_3)
            fs5_3 = self.refine5_8(F.upsample_bilinear(fs5_2, pool_x.size()[2:]) + gcfm5_4)
            fs5_4 = self.refine5_9(F.upsample_bilinear(fs5_3, conv_x.size()[2:]) + gcfm5_5)
            pred_5_lmk = self.refine5_10(F.upsample_bilinear(fs5_4, input.size()[2:]))

            return pred_1_lmk, pred_2_lmk, pred_3_lmk, pred_4_lmk, pred_5_lmk

        else:
            if method == 'KidneyLong':
                gcfm1_1 = self.refine1_1(self.gcn1_1(fm4))
                gcfm1_2 = self.refine1_2(self.gcn1_2(fm3))
                gcfm1_3 = self.refine1_3(self.gcn1_3(fm2))
                gcfm1_4 = self.refine1_4(self.gcn1_4(pool_x))
                gcfm1_5 = self.refine1_5(self.gcn1_5(conv_x))

                fs1_1 = self.refine1_6(F.upsample_bilinear(gcfm1_1, fm3.size()[2:]) + gcfm1_2)
                fs1_2 = self.refine1_7(F.upsample_bilinear(fs1_1, fm2.size()[2:]) + gcfm1_3)
                fs1_3 = self.refine1_8(F.upsample_bilinear(fs1_2, pool_x.size()[2:]) + gcfm1_4)
                fs1_4 = self.refine1_9(F.upsample_bilinear(fs1_3, conv_x.size()[2:]) + gcfm1_5)
                pred_1_lmk = self.refine1_10(F.upsample_bilinear(fs1_4, input.size()[2:]))
                return pred_1_lmk

            elif method == 'KidneyTrans':
                gcfm2_1 = self.refine2_1(self.gcn2_1(fm4))
                gcfm2_2 = self.refine2_2(self.gcn2_2(fm3))
                gcfm2_3 = self.refine2_3(self.gcn2_3(fm2))
                gcfm2_4 = self.refine2_4(self.gcn2_4(pool_x))
                gcfm2_5 = self.refine2_5(self.gcn2_5(conv_x))

                fs2_1 = self.refine2_6(F.upsample_bilinear(gcfm2_1, fm3.size()[2:]) + gcfm2_2)
                fs2_2 = self.refine2_7(F.upsample_bilinear(fs2_1, fm2.size()[2:]) + gcfm2_3)
                fs2_3 = self.refine2_8(F.upsample_bilinear(fs2_2, pool_x.size()[2:]) + gcfm2_4)
                fs2_4 = self.refine2_9(F.upsample_bilinear(fs2_3, conv_x.size()[2:]) + gcfm2_5)
                pred_2_lmk = self.refine2_10(F.upsample_bilinear(fs2_4, input.size()[2:]))
                return pred_2_lmk

            elif method == 'LiverLong':
                gcfm3_1 = self.refine3_1(self.gcn3_1(fm4))
                gcfm3_2 = self.refine3_2(self.gcn3_2(fm3))
                gcfm3_3 = self.refine3_3(self.gcn3_3(fm2))
                gcfm3_4 = self.refine3_4(self.gcn3_4(pool_x))
                gcfm3_5 = self.refine3_5(self.gcn3_5(conv_x))

                fs3_1 = self.refine3_6(F.upsample_bilinear(gcfm3_1, fm3.size()[2:]) + gcfm3_2)
                fs3_2 = self.refine3_7(F.upsample_bilinear(fs3_1, fm2.size()[2:]) + gcfm3_3)
                fs3_3 = self.refine3_8(F.upsample_bilinear(fs3_2, pool_x.size()[2:]) + gcfm3_4)
                fs3_4 = self.refine3_9(F.upsample_bilinear(fs3_3, conv_x.size()[2:]) + gcfm3_5)
                pred_3_lmk = self.refine3_10(F.upsample_bilinear(fs3_4, input.size()[2:]))
                return pred_3_lmk

            elif method == 'SpleenLong':
                gcfm4_1 = self.refine4_1(self.gcn4_1(fm4))
                gcfm4_2 = self.refine4_2(self.gcn4_2(fm3))
                gcfm4_3 = self.refine4_3(self.gcn4_3(fm2))
                gcfm4_4 = self.refine4_4(self.gcn4_4(pool_x))
                gcfm4_5 = self.refine4_5(self.gcn4_5(conv_x))

                fs4_1 = self.refine4_6(F.upsample_bilinear(gcfm4_1, fm3.size()[2:]) + gcfm4_2)
                fs4_2 = self.refine4_7(F.upsample_bilinear(fs4_1, fm2.size()[2:]) + gcfm4_3)
                fs4_3 = self.refine4_8(F.upsample_bilinear(fs4_2, pool_x.size()[2:]) + gcfm4_4)
                fs4_4 = self.refine4_9(F.upsample_bilinear(fs4_3, conv_x.size()[2:]) + gcfm4_5)
                pred_4_lmk = self.refine4_10(F.upsample_bilinear(fs4_4, input.size()[2:]))
                return pred_4_lmk

            elif method == 'SpleenTrans':
                gcfm5_1 = self.refine5_1(self.gcn5_1(fm4))
                gcfm5_2 = self.refine5_2(self.gcn5_2(fm3))
                gcfm5_3 = self.refine5_3(self.gcn5_3(fm2))
                gcfm5_4 = self.refine5_4(self.gcn5_4(pool_x))
                gcfm5_5 = self.refine5_5(self.gcn5_5(conv_x))

                fs5_1 = self.refine5_6(F.upsample_bilinear(gcfm5_1, fm3.size()[2:]) + gcfm5_2)
                fs5_2 = self.refine5_7(F.upsample_bilinear(fs5_1, fm2.size()[2:]) + gcfm5_3)
                fs5_3 = self.refine5_8(F.upsample_bilinear(fs5_2, pool_x.size()[2:]) + gcfm5_4)
                fs5_4 = self.refine5_9(F.upsample_bilinear(fs5_3, conv_x.size()[2:]) + gcfm5_5)
                pred_5_lmk = self.refine5_10(F.upsample_bilinear(fs5_4, input.size()[2:]))
                return pred_5_lmk



        # return out, fs4, fs3, fs2, fs1, gcfm1