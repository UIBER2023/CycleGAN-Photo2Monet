import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """残差块：CycleGAN生成器的核心组件"""
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """CycleGAN生成器网络（ResNet架构）"""
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9, n_downsampling=2):
        super(Generator, self).__init__()
        
        # 初始卷积层
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # 下采样层
        in_features = 64
        out_features = in_features * 2
        for _ in range(n_downsampling):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # 残差块
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # 上采样层
        out_features = in_features // 2
        for _ in range(n_downsampling):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, 
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # 输出卷积层
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """CycleGAN判别器网络（PatchGAN）"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(Discriminator, self).__init__()
        
        # 第一层不使用归一化
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 中间层
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                          kernel_size=4, stride=2, padding=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # 最后一层
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                      kernel_size=4, stride=1, padding=1),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 输出层
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    """重放缓冲区：用于稳定判别器训练"""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        to_return = []
        for element in data:
            element = element.detach()
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if torch.rand(1).item() > 0.5:
                    i = torch.randint(0, self.max_size, (1,)).item()
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.stack(to_return)