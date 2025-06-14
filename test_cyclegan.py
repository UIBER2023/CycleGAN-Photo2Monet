import torch
import os
from tqdm import tqdm
import time

from models import Generator
from loader import get_dataloader
from utils import save_image, tensor_to_pil

class CycleGANTester:
    """CycleGAN测试器类"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'photo2monet'), exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'monet2photo'), exist_ok=True)
        
        # 加载模型
        self._load_models()
        
        # 加载测试数据
        self.test_loader = get_dataloader(
            config['data_root'],
            mode='test',
            batch_size=1,  # 测试时使用batch_size=1
            shuffle=False,
            num_workers=config['num_workers']
        )
    
    def _load_models(self):
        """加载训练好的生成器模型"""
        # 初始化生成器
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
        
        # 加载权重
        print(f"Loading G_AB from {self.config['model_path_G_AB']}")
        self.G_AB.load_state_dict(
            torch.load(self.config['model_path_G_AB'], map_location=self.device)
        )
        
        print(f"Loading G_BA from {self.config['model_path_G_BA']}")
        self.G_BA.load_state_dict(
            torch.load(self.config['model_path_G_BA'], map_location=self.device)
        )
        
        # 设置为评估模式
        self.G_AB.eval()
        self.G_BA.eval()
    
    def test(self):
        """执行测试"""
        print("Starting testing...")
        start_time = time.time()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader, desc='Testing')):
                real_A = batch['A'].to(self.device)
                real_B = batch['B'].to(self.device)
                
                # 生成假图像
                fake_B = self.G_AB(real_A)
                fake_A = self.G_BA(real_B)
                
                # 保存结果
                # Photo to Monet
                save_image(
                    fake_B[0],
                    os.path.join(
                        self.config['output_dir'],
                        'photo2monet',
                        f'fake_monet_{i:04d}.png'
                    )
                )
                
                # Monet to Photo
                save_image(
                    fake_A[0],
                    os.path.join(
                        self.config['output_dir'],
                        'monet2photo',
                        f'fake_photo_{i:04d}.png'
                    )
                )
                
                # 可选：保存原始图像
                if self.config.get('save_originals', False):
                    save_image(
                        real_A[0],
                        os.path.join(
                            self.config['output_dir'],
                            'photo2monet',
                            f'real_photo_{i:04d}.png'
                        )
                    )
                    save_image(
                        real_B[0],
                        os.path.join(
                            self.config['output_dir'],
                            'monet2photo',
                            f'real_monet_{i:04d}.png'
                        )
                    )
        
        test_time = time.time() - start_time
        print(f"Testing completed in {test_time:.2f} seconds")
        print(f"Results saved to: {self.config['output_dir']}")
    
    def test_single_image(self, image_path, direction='AtoB'):
        """测试单张图像"""
        from PIL import Image
        from transforms import get_transform
        
        # 加载并预处理图像
        transform = get_transform(train=False)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        
        # 选择生成器
        generator = self.G_AB if direction == 'AtoB' else self.G_BA
        
        # 生成结果
        with torch.no_grad():
            fake_image = generator(image)
        
        # 转换为PIL图像
        result = tensor_to_pil(fake_image[0])
        
        # 保存结果
        output_name = f"{direction}_{os.path.basename(image_path)}"
        output_path = os.path.join(self.config['output_dir'], output_name)
        result.save(output_path)
        
        print(f"Result saved to: {output_path}")
        return result


def main():
    """主测试函数"""
    config = {
        # 数据参数
        'data_root': './monet2photo',
        'num_workers': 4,
        
        # 模型参数
        'n_residual_blocks': 9,
        'model_path_G_AB': './output_cyclegan/final_G_AB.pth',
        'model_path_G_BA': './output_cyclegan/final_G_BA.pth',
        
        # 输出参数
        'output_dir': './test_results',
        'save_originals': True,  # 是否保存原始图像
    }
    
    # 创建测试器
    tester = CycleGANTester(config)
    
    # 执行批量测试
    tester.test()
    
    # 测试单张图像示例（取消注释以使用）
    # result = tester.test_single_image(
    #     'path/to/your/image.jpg',
    #     direction='AtoB'  # 'AtoB' for photo to monet, 'BtoA' for monet to photo
    # )


if __name__ == "__main__":
    main()