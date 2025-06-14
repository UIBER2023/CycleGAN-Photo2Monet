import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
import os
import time
from tqdm import tqdm
import numpy as np

from models import Generator, Discriminator, ReplayBuffer
from loader import get_dataloader
from utils import save_image, init_weights, visualize_results

class CycleGANTrainer:
    """CycleGAN训练器类"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'samples'), exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
        
        # 初始化模型
        self._init_models()
        
        # 初始化损失函数
        self._init_losses()
        
        # 初始化优化器
        self._init_optimizers()
        
        # 初始化学习率调度器
        self._init_schedulers()
        
        # 初始化重放缓冲区
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
        
        # 初始化数据加载器
        self.train_loader = get_dataloader(
            config['data_root'], 
            mode='train', 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=config['num_workers']
        )
        
        # 训练状态
        self.start_epoch = 0
        self.global_step = 0
        
        # 如果需要，加载检查点
        if config.get('resume_checkpoint'):
            self.load_checkpoint(config['resume_checkpoint'])
    
    def _init_models(self):
        """初始化生成器和判别器"""
        # 生成器
        self.G_AB = Generator(
            input_nc=3, 
            output_nc=3, 
            n_residual_blocks=self.config['n_residual_blocks']
        ).to(self.device)
        self.G_BA = Generator(
            input_nc=3, 
            output_nc=3, 
            n_residual_blocks=self.config['n_residual_blocks']
        ).to(self.device)
        
        # 判别器
        self.D_A = Discriminator(input_nc=3).to(self.device)
        self.D_B = Discriminator(input_nc=3).to(self.device)
        
        # 初始化权重
        init_weights(self.G_AB, init_type=self.config['init_type'])
        init_weights(self.G_BA, init_type=self.config['init_type'])
        init_weights(self.D_A, init_type=self.config['init_type'])
        init_weights(self.D_B, init_type=self.config['init_type'])
    
    def _init_losses(self):
        """初始化损失函数"""
        self.criterion_GAN = nn.MSELoss()  # LSGAN损失
        self.criterion_cycle = nn.L1Loss()  # 循环一致性损失
        self.criterion_identity = nn.L1Loss()  # 身份损失
    
    def _init_optimizers(self):
        """初始化优化器"""
        # 生成器优化器
        self.optimizer_G = optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=self.config['lr'],
            betas=(self.config['beta1'], 0.999)
        )
        
        # 判别器优化器
        self.optimizer_D_A = optim.Adam(
            self.D_A.parameters(),
            lr=self.config['lr'],
            betas=(self.config['beta1'], 0.999)
        )
        self.optimizer_D_B = optim.Adam(
            self.D_B.parameters(),
            lr=self.config['lr'],
            betas=(self.config['beta1'], 0.999)
        )
    
    def _init_schedulers(self):
        """初始化学习率调度器"""
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - self.config['n_epochs']) / float(self.config['decay_epochs'] + 1)
            return lr_l
        
        self.scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=lambda_rule
        )
        self.scheduler_D_A = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=lambda_rule
        )
        self.scheduler_D_B = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=lambda_rule
        )
    
    def create_target_tensor(self, tensor_like, target_is_real):
        """
        动态创建目标张量，匹配输入张量的尺寸
        
        Args:
            tensor_like: 参考张量（通常是判别器输出）
            target_is_real: 是否为真实标签
            
        Returns:
            torch.Tensor: 目标张量
        """
        if target_is_real:
            target_tensor = torch.ones_like(tensor_like, requires_grad=False)
        else:
            target_tensor = torch.zeros_like(tensor_like, requires_grad=False)
        return target_tensor
    
    def train(self):
        """主训练循环"""
        print("Starting CycleGAN training...")
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config['n_epochs'] + self.config['decay_epochs']):
            epoch_start = time.time()
            
            # 训练一个epoch
            losses = self._train_epoch(epoch)
            
            # 更新学习率
            self.scheduler_G.step()
            self.scheduler_D_A.step()
            self.scheduler_D_B.step()
            
            # 打印epoch统计信息
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch [{epoch+1}/{self.config['n_epochs'] + self.config['decay_epochs']}] "
                  f"completed in {epoch_time:.2f}s")
            print(f"  Loss_G: {losses['G']:.4f}, Loss_D: {losses['D']:.4f}")
            print(f"  Loss_cycle: {losses['cycle']:.4f}, Loss_identity: {losses['identity']:.4f}")
            print(f"  Learning rate: {self.scheduler_G.get_last_lr()[0]:.6f}")
            
            # 保存样本图像
            if (epoch + 1) % self.config['sample_interval'] == 0:
                self.save_samples(epoch + 1)
            
            # 保存检查点
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(epoch + 1)
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        
        # 保存最终模型
        self.save_final_models()
    
    def _train_epoch(self, epoch):
        """训练一个epoch"""
        epoch_losses = {'G': 0, 'D': 0, 'cycle': 0, 'identity': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for i, batch in enumerate(pbar):
            real_A = batch['A'].to(self.device)
            real_B = batch['B'].to(self.device)
            
            # ========== 训练生成器 ==========
            self.G_AB.train()
            self.G_BA.train()
            self.optimizer_G.zero_grad()
            
            # 身份损失
            loss_identity_A = self.criterion_identity(self.G_BA(real_A), real_A)
            loss_identity_B = self.criterion_identity(self.G_AB(real_B), real_B)
            loss_identity = (loss_identity_A + loss_identity_B) * self.config['lambda_identity']
            
            # 生成假图像
            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)
            
            # GAN损失 - 动态创建目标张量
            pred_fake_B = self.D_B(fake_B)
            valid_B = self.create_target_tensor(pred_fake_B, True)
            loss_GAN_AB = self.criterion_GAN(pred_fake_B, valid_B)
            
            pred_fake_A = self.D_A(fake_A)
            valid_A = self.create_target_tensor(pred_fake_A, True)
            loss_GAN_BA = self.criterion_GAN(pred_fake_A, valid_A)
            
            loss_GAN = loss_GAN_AB + loss_GAN_BA
            
            # 循环一致性损失
            recov_A = self.G_BA(fake_B)
            loss_cycle_A = self.criterion_cycle(recov_A, real_A)
            
            recov_B = self.G_AB(fake_A)
            loss_cycle_B = self.criterion_cycle(recov_B, real_B)
            
            loss_cycle = (loss_cycle_A + loss_cycle_B) * self.config['lambda_cycle']
            
            # 总生成器损失
            loss_G = loss_GAN + loss_cycle + loss_identity
            loss_G.backward()
            self.optimizer_G.step()
            
            # ========== 训练判别器A ==========
            self.optimizer_D_A.zero_grad()
            
            # 真实损失
            pred_real_A = self.D_A(real_A)
            valid_real_A = self.create_target_tensor(pred_real_A, True)
            loss_real_A = self.criterion_GAN(pred_real_A, valid_real_A)
            
            # 假损失（使用重放缓冲区）
            fake_A_ = self.fake_A_buffer.push_and_pop(fake_A.detach())
            pred_fake_A_ = self.D_A(fake_A_)
            fake_fake_A = self.create_target_tensor(pred_fake_A_, False)
            loss_fake_A = self.criterion_GAN(pred_fake_A_, fake_fake_A)
            
            # 总判别器A损失
            loss_D_A = (loss_real_A + loss_fake_A) * 0.5
            loss_D_A.backward()
            self.optimizer_D_A.step()
            
            # ========== 训练判别器B ==========
            self.optimizer_D_B.zero_grad()
            
            # 真实损失
            pred_real_B = self.D_B(real_B)
            valid_real_B = self.create_target_tensor(pred_real_B, True)
            loss_real_B = self.criterion_GAN(pred_real_B, valid_real_B)
            
            # 假损失（使用重放缓冲区）
            fake_B_ = self.fake_B_buffer.push_and_pop(fake_B.detach())
            pred_fake_B_ = self.D_B(fake_B_)
            fake_fake_B = self.create_target_tensor(pred_fake_B_, False)
            loss_fake_B = self.criterion_GAN(pred_fake_B_, fake_fake_B)
            
            # 总判别器B损失
            loss_D_B = (loss_real_B + loss_fake_B) * 0.5
            loss_D_B.backward()
            self.optimizer_D_B.step()
            
            loss_D = loss_D_A + loss_D_B
            
            # 更新进度条
            pbar.set_postfix({
                'G': f'{loss_G.item():.4f}',
                'D': f'{loss_D.item():.4f}',
                'cyc': f'{loss_cycle.item():.4f}'
            })
            
            # 累积损失
            epoch_losses['G'] += loss_G.item()
            epoch_losses['D'] += loss_D.item()
            epoch_losses['cycle'] += loss_cycle.item()
            epoch_losses['identity'] += loss_identity.item()
            
            self.global_step += 1
        
        # 计算平均损失
        n_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def save_samples(self, epoch):
        """保存样本图像"""
        self.G_AB.eval()
        self.G_BA.eval()
        
        with torch.no_grad():
            # 获取一个批次的数据
            batch = next(iter(self.train_loader))
            real_A = batch['A'].to(self.device)
            real_B = batch['B'].to(self.device)
            
            # 生成假图像
            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)
            
            # 保存可视化结果
            for i in range(min(4, real_A.size(0))):
                visualize_results(
                    real_A[i], real_B[i], fake_A[i], fake_B[i],
                    save_path=os.path.join(
                        self.config['output_dir'], 
                        'samples', 
                        f'epoch_{epoch}_sample_{i}.png'
                    )
                )
    
    def save_checkpoint(self, epoch):
        """保存训练检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'G_AB_state_dict': self.G_AB.state_dict(),
            'G_BA_state_dict': self.G_BA.state_dict(),
            'D_A_state_dict': self.D_A.state_dict(),
            'D_B_state_dict': self.D_B.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_A_state_dict': self.optimizer_D_A.state_dict(),
            'optimizer_D_B_state_dict': self.optimizer_D_B.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_A_state_dict': self.scheduler_D_A.state_dict(),
            'scheduler_D_B_state_dict': self.scheduler_D_B.state_dict(),
        }
        
        path = os.path.join(
            self.config['output_dir'], 
            'checkpoints', 
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载训练检查点"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        self.G_AB.load_state_dict(checkpoint['G_AB_state_dict'])
        self.G_BA.load_state_dict(checkpoint['G_BA_state_dict'])
        self.D_A.load_state_dict(checkpoint['D_A_state_dict'])
        self.D_B.load_state_dict(checkpoint['D_B_state_dict'])
        
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
        self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
        
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D_A.load_state_dict(checkpoint['scheduler_D_A_state_dict'])
        self.scheduler_D_B.load_state_dict(checkpoint['scheduler_D_B_state_dict'])
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def save_final_models(self):
        """保存最终模型"""
        torch.save(self.G_AB.state_dict(), 
                   os.path.join(self.config['output_dir'], 'final_G_AB.pth'))
        torch.save(self.G_BA.state_dict(), 
                   os.path.join(self.config['output_dir'], 'final_G_BA.pth'))
        torch.save(self.D_A.state_dict(), 
                   os.path.join(self.config['output_dir'], 'final_D_A.pth'))
        torch.save(self.D_B.state_dict(), 
                   os.path.join(self.config['output_dir'], 'final_D_B.pth'))
        print("Final models saved!")


if __name__ == "__main__":
    # 训练配置
    config = {
        # 数据参数
        'data_root': './monet2photo',
        'batch_size': 4,
        'num_workers': 4,
        
        # 模型参数
        'n_residual_blocks': 9,  # ResNet块数量
        'init_type': 'normal',
        
        # 训练参数
        'n_epochs': 100,         # 固定学习率的epoch数
        'decay_epochs': 100,     # 学习率衰减的epoch数
        'lr': 0.0002,
        'beta1': 0.5,
        
        # 损失权重
        'lambda_cycle': 10.0,    # 循环一致性损失权重
        'lambda_identity': 5.0,  # 身份损失权重
        
        # 保存参数
        'output_dir': './output_cyclegan',
        'sample_interval': 5,      # 每5个epoch保存样本
        'checkpoint_interval': 10, # 每10个epoch保存检查点
        
        # 恢复训练（可选）
        'resume_checkpoint': None,  # 设置检查点路径以恢复训练
    }
    
    # 创建训练器并开始训练
    trainer = CycleGANTrainer(config)
    trainer.train()