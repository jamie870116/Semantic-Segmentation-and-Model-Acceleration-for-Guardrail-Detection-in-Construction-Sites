import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils

class DownsampleBlock(nn.Module):
    def __init__(self, nc_input, nc_output):
        '''
        Arguments:
        nc_input : Win, number of input channel
        nc_output : Wout, number of output channel
        '''

        super(DownsampleBlock,self).__init__()
        self.nc_input = nc_input
        self.nc_output = nc_output

        if self.nc_input < self.nc_output:
          # Win < Wout
            self.conv = nn.Conv2d(nc_input, nc_output-nc_input, kernel_size=3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
          # Win > Wout
            self.conv = nn.Conv2d(nc_input, nc_output, kernel_size=3, stride=2, padding=1)

        self.batchNorm = nn.BatchNorm2d(nc_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.nc_input < self.nc_output:
            out = torch.cat([self.conv(x), self.pool(x)], 1)
        else:
            out = self.conv(x)

        out = self.batchNorm(out)
        out = self.relu(out)
        return out

class EDABlock(nn.Module):
    def __init__(self, nc_input, dilated, k = 40, dropprob = 0.02):
        '''
        Arguments:
        nc_input : number of input channel
        k : growth rate
        dilated : possible dilated convalution
        dropprob : probability, a dropout layer between the last ReLU and the concatenation of each module
        '''
        super(EDABlock,self).__init__()
        self.conv1x1_0 = nn.Conv2d(nc_input, k, kernel_size=1)
        self.batchNorm_0 = nn.BatchNorm2d(k)

        self.conv3x1_1 = nn.Conv2d(k, k, kernel_size=(3,1), padding=(1,0))
        self.conv1x3_1 = nn.Conv2d(k, k, kernel_size=(1,3), padding=(0,1))
        self.batchNorm_1 = nn.BatchNorm2d(k)

        self.conv3x1_2 = nn.Conv2d(k, k, kernel_size=(3,1), stride=1, padding=(dilated,0), dilation=dilated)
        self.conv1x3_2 = nn.Conv2d(k, k, kernel_size=(1,3), stride=1, padding=(0,dilated), dilation=dilated)
        self.batchNorm_2 = nn.BatchNorm2d(k)
        self.dropout = nn.Dropout2d(dropprob)
        self.relu = nn.ReLU()

    def forward(self, x):
        input = x

        output = self.conv1x1_0(x)
        output = self.batchNorm_0(output)
        output = self.relu(output)

        output = self.conv3x1_1(output)
        output = self.conv1x3_1(output)
        output = self.batchNorm_1(output)
        output = self.relu(output)

        output = self.conv3x1_2(output)
        output = self.conv1x3_2(output)
        output = self.batchNorm_2(output)
        output = self.relu(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        output = torch.cat((output, input), 1)

        return output

class EDAnet(nn.Module):
    def __init__(self, n_class=1):
        '''
        Arguments:
        nc_input : number of input channel
        k : growth rate
        dilated : possible dilated convalution
        dropprob : probability, a dropout layer between the last ReLU and the concatenation of each module
        '''
        super(EDAnet,self).__init__()
        self.layers = nn.ModuleList()
        self.dilation1 = [1,1,1,2,2]
        self.dilation2 = [2,2,4,4,8,8,16,16]

        # DownsampleBlock1
        self.layers.append(DownsampleBlock(3, 15))

        # DownsampleBlock2
        self.layers.append(DownsampleBlock(15, 60))

        # EDA module 1-1~1-5
        for i in range(len(self.dilation1)):
            self.layers.append(EDABlock(60 + 40 * i, self.dilation1[i]))

        # DownsampleBlock3
        self.layers.append(DownsampleBlock(260, 130))

        # EDA module 2-1~2-8
        for j in range(len(self.dilation2)):
            self.layers.append(EDABlock(130 + 40 * j, self.dilation2[j]))

        # Projection layer
        self.project_layer = nn.Conv2d(450, n_class, kernel_size = 1)

        self.weights_init()

    def weights_init(self):
        for index, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        output = x

        for layer in self.layers:
            output = layer(output)
          # print(output.shape)

        output = self.project_layer(output)

        # Bilinear interpolation x8
        output = F.interpolate(output,scale_factor = 8,mode = 'bilinear',align_corners=True)

        # # Bilinear interpolation x2 (inference only)
        # if not self.training:
        #   output = F.interpolate(output, scale_factor=2, mode='bilinear',align_corners=True)

        return output


if __name__=="__main__":
    
    edanet = EDAnet().eval()
    device = torch.device("cuda")
    edanet.to(device)
    
    input =torch.randn(1,3,720,960).cuda()

    torch.cuda.synchronize()
    time_start = time.time()

    output = edanet(input)
    torch.cuda.synchronize()
    time_end = time.time()
    infer_time = time_end - time_start
    print(infer_time)
