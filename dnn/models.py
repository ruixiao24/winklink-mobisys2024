import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

n_channel = 3
## PRN
class PRN(nn.Module):
    def __init__(self, recurrent_iter=5, use_GPU=True):
        super(PRN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(2*n_channel, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, n_channel, 3, 1, 1),
        )

    def forward(self, input):

        x = input

        for _ in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input

        return x



class ExtractStripeSubmodel(nn.Module):

    def __init__(self):
        super(ExtractStripeSubmodel, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(2*n_channel, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
        )

        self.conv7 = nn.Sequential(nn.Conv2d(32, 32, 3, (2, 1), 1), nn.BatchNorm2d(32), nn.ReLU(),
                                    nn.Conv2d(32, 1, 3, (2, 1), 1), nn.Tanh())

    def forward(self, x):
        x = self.conv0(x)
        x = F.max_pool2d(x, kernel_size=(2, 1))
        resx = x
        x = F.relu(self.res_conv1(x) + resx)
        x = F.max_pool2d(x, kernel_size=(2, 1))
        resx = x
        x = F.relu(self.res_conv2(x) + resx)
        x = F.max_pool2d(x, kernel_size=(2, 1))
        resx = x
        x = F.relu(self.res_conv3(x) + resx)
        x = F.max_pool2d(x, kernel_size=(2, 1))
        resx = x
        x = F.relu(self.res_conv4(x) + resx)
        x = F.max_pool2d(x, kernel_size=(2, 1))
        resx = x
        x = F.relu(self.res_conv5(x) + resx)
        x = F.max_pool2d(x, kernel_size=(2, 1))
        
        x = self.conv6(x)
        x = F.max_pool2d(x, kernel_size=(2, 1))

        x = self.conv7(x)      # SHAPE: [B,  1,  1, Width]

        return x.squeeze(2)


if __name__ == "__main__":

    model = ExtractStripeSubmodel()
