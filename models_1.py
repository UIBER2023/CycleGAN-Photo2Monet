import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.optim as optim

# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)  # 残差连接
你好啦啦啦啦啦啦
# 生成器定义
class Generator(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(Generator, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 残差块 (9个)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks)]
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

# 普通判别器定义 (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# 带有谱归一化的判别器定义 (CycleGAN-SN)
class DiscriminatorSN(nn.Module):
    def __init__(self):
        super(DiscriminatorSN, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))
        )
    
    def forward(self, x):
        return self.model(x)

# 损失函数定义
def adversarial_loss(pred, target):
    """对抗损失：使用均方误差 (MSE)"""
    return nn.MSELoss()(pred, target)

def cycle_consistency_loss(real, reconstructed):
    """循环一致性损失：使用L1损失"""
    return nn.L1Loss()(real, reconstructed)

def identity_loss(real, identity):
    """身份损失：使用L1损失 (可选)"""
    return nn.L1Loss()(real, identity)

# 示例训练循环框架
def train_cycle_gan(dataloader, num_epochs=200):
    # 实例化模型
    gen_A2B = Generator().cuda()
    gen_B2A = Generator().cuda()
    disc_A = DiscriminatorSN().cuda()  # 使用带谱归一化的判别器
    disc_B = DiscriminatorSN().cuda()

    # 优化器
    gen_optimizer = optim.Adam(
        list(gen_A2B.parameters()) + list(gen_B2A.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    disc_optimizer = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )

    # 超参数
    lambda_cycle = 10.0  # 循环一致性损失权重
    lambda_identity = 0.3  # 身份损失权重 (根据你的要求调整)

    # 训练循环
    for epoch in range(num_epochs):
        for real_A, real_B in dataloader:
            real_A, real_B = real_A.cuda(), real_B.cuda()

            # 判别器训练
            disc_optimizer.zero_grad()
            fake_B = gen_A2B(real_A)
            fake_A = gen_B2A(real_B)
            
            disc_loss_A = (adversarial_loss(disc_A(real_A), torch.ones_like(disc_A(real_A))) +
                           adversarial_loss(disc_A(fake_A.detach()), torch.zeros_like(disc_A(fake_A))))
            disc_loss_B = (adversarial_loss(disc_B(real_B), torch.ones_like(disc_B(real_B))) +
                           adversarial_loss(disc_B(fake_B.detach()), torch.zeros_like(disc_B(fake_B))))
            
            disc_loss = (disc_loss_A + disc_loss_B) * 0.5
            disc_loss.backward()
            disc_optimizer.step()

            # 生成器训练
            gen_optimizer.zero_grad()
            
            # 对抗损失
            adv_loss_A2B = adversarial_loss(disc_B(fake_B), torch.ones_like(disc_B(fake_B)))
            adv_loss_B2A = adversarial_loss(disc_A(fake_A), torch.ones_like(disc_A(fake_A)))
            
            # 循环一致性损失
            cycle_A = gen_B2A(fake_B)
            cycle_B = gen_A2B(fake_A)
            cycle_loss = (cycle_consistency_loss(real_A, cycle_A) +
                          cycle_consistency_loss(real_B, cycle_B))
            
            # 身份损失 (可选)
            identity_A = gen_B2A(real_A)
            identity_B = gen_A2B(real_B)
            id_loss = (identity_loss(real_A, identity_A) +
                       identity_loss(real_B, identity_B))
            
            # 总生成器损失
            gen_loss = adv_loss_A2B + adv_loss_B2A + lambda_cycle * cycle_loss + lambda_identity * id_loss
            gen_loss.backward()
            gen_optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")

# 单元测试
if __name__ == "__main__":
    # 测试生成器
    gen = Generator()
    x = torch.randn(2, 3, 256, 256)
    y = gen(x)
    print("Generator output shape:", y.shape)  # 应为 [2, 3, 256, 256]

    # 测试判别器 (带谱归一化)
    disc = DiscriminatorSN()
    y = disc(x)
    print("DiscriminatorSN output shape:", y.shape)  # 应为 [2, 1, 16, 16]