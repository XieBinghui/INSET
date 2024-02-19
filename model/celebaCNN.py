import torch
import torch.nn as nn
import torch.nn.functional as F

class celebaCNN(nn.Sequential):
    def __init__(self, symmetry=False):
        super(celebaCNN, self).__init__()

        # in_ch = [3] + [32,64,128]
        # kernels = [3,4,5]
        # strides = [2,2,2]
        # layer_size = 4
        # self.symmetry = symmetry
        # self.layer_size = 4
        in_ch = [3] + [32,64,128]
        kernels = [3,4,5]
        strides = [2,2,2]
        layer_size = 3
        self.symmetry = symmetry
        self.layer_size = 3
        self.conv = nn.ModuleList([nn.Conv2d(in_channels = in_ch[i], 
                                                out_channels = in_ch[i+1], 
                                                kernel_size = kernels[i],
                                                stride = strides[i]) for i in range(layer_size)])
        self.conv2 = nn.ModuleList([nn.Conv2d(in_channels = in_ch[i], 
                                                out_channels = in_ch[i+1], 
                                                kernel_size = kernels[i],
                                                stride = strides[i]) for i in range(layer_size)]).double()
        self.pool = nn.ModuleList([nn.MaxPool2d(kernel_size = kernels[i],
                                                stride = strides[i]) for i in range(layer_size)]).double()
        
        # self.linear = nn.ModuleList([nn.Linear(31*31, 31*31),nn.Linear(14*14, 14*14),nn.Linear(5*5, 5*5)]).double()
        self.conv = self.conv.double()
        self.fc1 = nn.Linear(128, 256)
        self.bias = [nn.Parameter(torch.FloatTensor([1.0])).to('cuda'), nn.Parameter(torch.FloatTensor([1.0])).to('cuda'),nn.Parameter(torch.FloatTensor([1.0])).to('cuda')]

    def _forward_features(self, x):
        if self.symmetry:
            for i in range(self.layer_size):
                if i==2:
                    n, c, h, w = x.size()
                    bs = int(n/8)
                    x1 = torch.mean(x.reshape(bs,8,c,h,w), dim=1,keepdim=False)
                    x1 = self.pool[i](x1)
                    x2 = self.conv[i](x)
                    n, c, h, w = x2.size()
                    bs = int(n/8)
                    x1 = torch.mean(x1, dim=1,keepdim=True)
                    x1 = x1.unsqueeze(1).repeat(1,8,1,1,1).repeat(1,1,c,1,1)
                    x1 = F.elu(x1).view(n,c,h,w)
                    x = x2+x1
                    x = F.relu(x)
                else:
                    x = self.conv[i](x)
            # for i in range(self.layer_size):
            #     n, c, h, w = x.size()
            #     bs = int(n/8)
            #     x1 = torch.mean(x.reshape(bs,8,c,h,w), dim=1,keepdim=False)
            #     x1 = self.conv2[i](x1)
            #     x2 = self.conv[i](x)
            #     n, c, h, w = x2.size()
            #     bs = int(n/8)
            #     x1 = x1.unsqueeze(1).repeat(1,8,1,1,1).view(n,c,h,w)
            #     x = x2+x1
            #     x = F.relu(x)
        else:
            for l in self.conv:
                x = F.relu(l(x))
        x = F.adaptive_max_pool2d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v