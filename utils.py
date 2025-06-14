import torch
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def save_image(tensor, path, nrow=1):
    """
    将张量保存为图像文件
    
    Args:
        tensor: 输入张量，范围[-1, 1]
        path: 保存路径
        nrow: 每行显示的图像数（用于批量保存）
    """
    # 创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 将张量从[-1, 1]转换到[0, 1]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1).detach().cpu().numpy()
    
    # 转换维度顺序：CHW -> HWC
    tensor = np.transpose(tensor, (1, 2, 0)) * 255
    
    # 保存图像
    image = Image.fromarray(tensor.astype(np.uint8))
    image.save(path)

def tensor_to_pil(tensor):
    """
    将张量转换为PIL图像
    
    Args:
        tensor: 输入张量，范围[-1, 1]
        
    Returns:
        PIL.Image: PIL图像对象
    """
    # 将张量从[-1, 1]转换到[0, 1]
    tensor = (tensor + 1) / 2
    array = tensor.detach().cpu().numpy()
    
    # 转换维度顺序：CHW -> HWC
    array = np.transpose(array, (1, 2, 0)) * 255
    
    return Image.fromarray(array.astype(np.uint8))

def init_weights(net, init_type='normal', init_gain=0.02):
    """
    初始化网络权重
    
    Args:
        net: 网络模型
        init_type: 初始化类型 ('normal', 'xavier', 'kaiming', 'orthogonal')
        init_gain: 初始化增益
    """
    def init_func(m):
        classname = m.__class__.__name__
        
        # 卷积层和全连接层的初始化
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')
            
            # 偏置项初始化为0
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        
        # BatchNorm层的初始化
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)
        
        # InstanceNorm层的初始化
        elif classname.find('InstanceNorm2d') != -1:
            if m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    
    print(f'Initializing network with {init_type} initialization')
    net.apply(init_func)

def visualize_results(real_A, real_B, fake_A, fake_B, save_path=None):
    """
    可视化CycleGAN的结果
    
    Args:
        real_A: 真实的A域图像（照片）
        real_B: 真实的B域图像（莫奈画作）
        fake_A: 生成的A域图像
        fake_B: 生成的B域图像
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 显示图像
    images = [
        (real_A, 'Real Photo'),
        (fake_B, 'Generated Monet'),
        (real_B, 'Real Monet'),
        (fake_A, 'Generated Photo')
    ]
    
    for ax, (img, title) in zip(axes.flat, images):
        # 转换为numpy数组并调整范围
        if isinstance(img, torch.Tensor):
            img = tensor_to_pil(img)
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_stats(dataloader):
    """
    计算数据集的均值和标准差（用于数据标准化）
    
    Args:
        dataloader: 数据加载器
        
    Returns:
        tuple: (mean, std) 每个通道的均值和标准差
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for data in dataloader:
        batch_samples = data['A'].size(0)
        data_a = data['A'].view(batch_samples, data['A'].size(1), -1)
        data_b = data['B'].view(batch_samples, data['B'].size(1), -1)
        
        # 计算当前批次的统计信息
        mean += data_a.mean(2).sum(0) + data_b.mean(2).sum(0)
        std += data_a.std(2).sum(0) + data_b.std(2).sum(0)
        total_samples += batch_samples * 2  # A和B两个域
    
    mean /= total_samples
    std /= total_samples
    
    return mean, std