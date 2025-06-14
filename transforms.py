from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_transform(train=True):
    """
    获取数据变换管道
    
    Args:
        train (bool): 是否为训练模式
        
    Returns:
        transforms.Compose: 数据变换管道
    """
    transform_list = []
    
    if train:
        # 训练模式：先调整大小到286x286，然后随机裁剪到256x256
        transform_list.append(transforms.Resize((286, 286), interpolation=InterpolationMode.BICUBIC))
        transform_list.append(transforms.RandomCrop(256))
        # 随机水平翻转
        transform_list.append(transforms.RandomHorizontalFlip())
        # 随机旋转±15度
        transform_list.append(transforms.RandomRotation(15))
        # 颜色抖动：亮度±10%，对比度±15%
        transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.15))
    else:
        # 测试模式：直接调整到256x256
        transform_list.append(transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC))
    
    # 转换为张量
    transform_list.append(transforms.ToTensor())
    # 归一化到[-1, 1]范围
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    return transforms.Compose(transform_list)