from torch.utils.data import DataLoader
from datasets import Monet2PhotoDataset
from transforms import get_transform

def get_dataloader(root_dir, mode='train', batch_size=1, shuffle=True, num_workers=0):
    """
    创建数据加载器
    
    Args:
        root_dir (str): 数据集根目录
        mode (str): 'train' 或 'test'
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        num_workers (int): 工作进程数
        
    Returns:
        DataLoader: PyTorch数据加载器
    """
    dataset = Monet2PhotoDataset(
        root_dir=root_dir, 
        mode=mode, 
        transform=get_transform(train=(mode=='train'))
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )