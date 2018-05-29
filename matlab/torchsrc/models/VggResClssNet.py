import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VggResClssNet(nn.Module):

    def __init__(self, n_class=21):
        super(VggResClssNet, self).__init__()


        self.resdown = nn.Sequential(
            nn.Linear(8192, 1024),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
            nn.Dropout2d(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
            nn.Dropout2d(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
            nn.Dropout2d(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(1024, n_class),
        )

        self.fc4 = nn.Sequential(
            nn.Linear(8192+1024, 2048),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
            nn.Dropout2d(),
        )

        self.fc5= nn.Sequential(
            nn.Linear(2048, n_class),
        )

    def forward(self, x1, x2):

        # x2 = self.resdown(x2)
        x = torch.cat([x1, x2], 1)
        # x = self.fc1(x)
        # x = self.fc2(x)

        #x = torch.cat([x1, x2], 1)
        x = self.fc4(x)
        x = self.fc5(x)

        return F.log_softmax(x)

