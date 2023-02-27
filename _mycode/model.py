from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
        
class Inversion(nn.Module):
    def __init__(self, nc, ngf, nz, truncation, c,data_size):
        super(Inversion, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.truncation = truncation
        self.c = c

        if data_size == 64:
            self.decoder = nn.Sequential(
                # input is Z
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
                nn.BatchNorm2d(ngf * 8),
                nn.Tanh(),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
                nn.BatchNorm2d(ngf * 4),
                nn.Tanh(),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
                nn.BatchNorm2d(ngf * 2),
                nn.Tanh(),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
                nn.BatchNorm2d(ngf),
                nn.Tanh(),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
            )
        elif data_size == 128:
            self.decoder = nn.Sequential(
            # input is Z
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 16, kernel_size=4, stride=1, padding=0), #反卷积层
            nn.BatchNorm2d(ngf * 16),
            nn.Tanh(),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1),
            nn.BatchNorm2d(ngf * 1),
            nn.Tanh(),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 128 x 128
        )
        elif data_size == 256:
            self.decoder = nn.Sequential(
            # input is Z
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 32, kernel_size=4, stride=1, padding=0), #反卷积层
            nn.BatchNorm2d(ngf * 32),
            nn.Tanh(),
            # state size. (ngf*32) x 4 x 4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1),
            nn.BatchNorm2d(ngf * 16),
            nn.Tanh(),
            # state size. (ngf*16) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1),
            nn.BatchNorm2d(ngf * 1),
            nn.Tanh(),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 256 x 256
            )
        else:
            raise "img_size error"

    def forward(self, x):
        topk, indices = torch.topk(x, self.truncation) # 按维度求前k个最值以及其索引
        topk = torch.clamp(torch.log(topk), min=-1000) + self.c # 将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        # x = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, topk) #ANCHOR - 暂无gpu版本的torch，因此改了代码
        x = torch.zeros(len(x), self.nz,dtype=torch.float64).scatter_(1, indices, topk) #将src中所有的值分散到self 中，填法是按照index中所指示的索引来填入

        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x.float())
        return x
    
class DGWGAN(nn.Module):
    def __init__(self, in_dim=3, dim=128):
        super(DGWGAN, self).__init__()
        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))
    
    def forward(self, x):
        y = self.ls(x.float())
        y = y.view(-1)
        return y