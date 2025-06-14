import torch
import torch.nn as nn
import itertools
from loader import get_dataloader
from utils import save_image, init_weights
import os
import time

class SimpleGenerator(nn.Module):
    """
    简化版生成器网络
    注意：这是一个极简的实现，实际CycleGAN使用更复杂的ResNet架构
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 输入层
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 中间层（简化版）
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 上采样
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 输出层
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)

# ==== 训练配置 ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数
batch_size = 4
num_epochs = 5
learning_rate = 0.0002
root_path = './monet2photo'
output_dir = './output'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# ==== 数据加载 ====
print("Loading datasets...")
train_loader = get_dataloader(root_path, mode='train', batch_size=batch_size, shuffle=True)
print(f"Total training batches: {len(train_loader)}")

# ==== 模型初始化 ====
print("Initializing models...")
G_AB = SimpleGenerator().to(device)  # Photo to Monet
G_BA = SimpleGenerator().to(device)  # Monet to Photo

# 初始化权重
init_weights(G_AB)
init_weights(G_BA)

# ==== 损失函数和优化器 ====
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), 
    lr=learning_rate, 
    betas=(0.5, 0.999)
)

# ==== 训练循环 ====
print("Starting training...")
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    total_loss = 0
    
    for i, data in enumerate(train_loader):
        # 获取输入数据
        real_A = data['A'].to(device)  # 照片
        real_B = data['B'].to(device)  # 莫奈画作
        
        # 前向传播
        fake_B = G_AB(real_A)  # 照片 -> 莫奈风格
        fake_A = G_BA(real_B)  # 莫奈 -> 照片风格
        
        # 计算损失（简化版，实际CycleGAN包含循环一致性损失等）
        loss = criterion(fake_B, real_B) + criterion(fake_A, real_A)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 定期打印进度和保存样本
        if i % 50 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")
            
            # 保存生成的样本
            save_image(fake_B[0], f'{output_dir}/fake_monet_epoch{epoch+1}_batch{i}.png')
            save_image(fake_A[0], f'{output_dir}/fake_photo_epoch{epoch+1}_batch{i}.png')
            
            # 保存原始图像作为对比
            if epoch == 0 and i == 0:
                save_image(real_A[0], f'{output_dir}/real_photo_sample.png')
                save_image(real_B[0], f'{output_dir}/real_monet_sample.png')
    
    # 计算并打印epoch统计信息
    avg_loss = total_loss / len(train_loader)
    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s, "
          f"Average Loss: {avg_loss:.4f}")
    
    # 保存模型检查点
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch + 1,
            'G_AB_state_dict': G_AB.state_dict(),
            'G_BA_state_dict': G_BA.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'{output_dir}/checkpoint_epoch{epoch+1}.pth')
        print(f"Checkpoint saved at epoch {epoch+1}")

# 训练完成
total_time = time.time() - start_time
print(f"\nTraining completed in {total_time/60:.2f} minutes")
print(f"Generated images saved in: {output_dir}")

# 保存最终模型
torch.save(G_AB.state_dict(), f'{output_dir}/final_G_AB.pth')
torch.save(G_BA.state_dict(), f'{output_dir}/final_G_BA.pth')
print("Final models saved!")