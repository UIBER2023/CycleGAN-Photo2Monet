import os
import random
from PIL import Image
from torch.utils.data import Dataset

class Monet2PhotoDataset(Dataset):
    """
    莫奈画作与照片风格转换数据集
    
    数据集结构：
    - train_photo: 训练用普通照片（7028张）
    - train_monet: 训练用莫奈画作（820张）
    - test_photo: 测试用普通照片
    - test_monet: 测试用莫奈画作
    """
    
    def __init__(self, root_dir, mode='train', transform=None):
        """
        初始化数据集
        
        Args:
            root_dir (str): 数据集根目录
            mode (str): 'train' 或 'test'
            transform: 数据变换管道
        """
        self.transform = transform
        
        # 设置域A（照片）和域B（莫奈画作）的目录
        if mode == 'train':
            self.dir_a = os.path.join(root_dir, "train_photo")
            self.dir_b = os.path.join(root_dir, "train_monet")
        else:
            self.dir_a = os.path.join(root_dir, "test_photo")
            self.dir_b = os.path.join(root_dir, "test_monet")
        
        # 加载所有图像文件路径
        self.domain_a = sorted([
            os.path.join(self.dir_a, f) for f in os.listdir(self.dir_a)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.domain_b = sorted([
            os.path.join(self.dir_b, f) for f in os.listdir(self.dir_b)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        self.len_a = len(self.domain_a)
        self.len_b = len(self.domain_b)
        # 数据集长度取两个域的最大值
        self.length = max(self.len_a, self.len_b)
        
        print(f"Loaded dataset in {mode} mode:")
        print(f"  Domain A (photos): {self.len_a} images")
        print(f"  Domain B (monet): {self.len_b} images")
    
    def __getitem__(self, index):
        """
        获取一对图像（照片和莫奈画作）
        
        Args:
            index (int): 索引
            
        Returns:
            dict: 包含'A'（照片）和'B'（莫奈画作）的字典
        """
        # 域A使用循环索引
        a_index = index % self.len_a
        # 域B使用随机索引，实现不配对的训练
        b_index = random.randint(0, self.len_b - 1)
        
        # 加载图像并转换为RGB格式
        image_a = Image.open(self.domain_a[a_index]).convert('RGB')
        image_b = Image.open(self.domain_b[b_index]).convert('RGB')
        
        # 应用数据变换
        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)
        
        return {'A': image_a, 'B': image_b}
    
    def __len__(self):
        return self.length