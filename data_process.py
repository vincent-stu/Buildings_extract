from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
from rasterio.errors import NotGeoreferencedWarning
from config_loader import load_config
import warnings
import os
import numpy as np
import torch
import random

# 抑制NotGeoreferencedWarning警告
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

config = load_config()

def read_img(img_path):
    """
    读取图像, 并将其guard到[0,1]之间
    """
    with rasterio.open(img_path) as src:
        image = src.read()
        image = image.astype(np.float32) / 255.0 
    
    return image

def read_label(label_path):
    """
    读取标签, 将255转换为1，其他值转换为0
    """
    with rasterio.open(label_path) as src:
        label = src.read(1)
        label = (label == 255).astype(np.int64)
    
    return label

def data_enhance(img, label):
    """
    数据增强
    """
    hor = random.choice([True, False])  # 随机选择是否进行水平翻转
    if hor:
        img = np.ascontiguousarray(np.flip(img, axis = 2)) # 水平翻转
        label = np.ascontiguousarray(np.flip(label, axis = 1)) # 水平翻转
    ver = random.choice([True, False])  # 随机选择是否进行垂直翻转
    if ver:
        img = np.ascontiguousarray(np.flip(img, axis = 1)) # 垂直翻转
        label = np.ascontiguousarray(np.flip(label, axis = 0)) # 垂直翻转
    
    return img, label

# 定义数据集类
class BuildingDataset(Dataset):
    def __init__(self, image_dir, label_dir, mode='train'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.data_clean(image_dir, label_dir)  # 清理无效文件
        self.mode = mode
        self.images = os.listdir(image_dir)
        self.image_files = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(".tiff")]
        self.label_files = [os.path.join(label_dir, fname) for fname in os.listdir(label_dir) if fname.endswith(".tif")]
    
    @staticmethod
    def data_clean(data_dir, label_dir):
        """
        清理数据集，删除无效文件
        """
        for filename in os.listdir(data_dir):
            if filename.endswith(".tiff"):
                pass
            else:
                os.remove(os.path.join(data_dir, filename))
                
        for filename in os.listdir(label_dir):
            if filename.endswith(".tif"):
                pass
            else:
                os.remove(os.path.join(label_dir, filename))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.tiff', '.tif'))

        # 读取图像和标签
        image = read_img(img_path)
        label = read_label(label_path)

        # 数据增强
        if self.mode == "train":
            image, label = data_enhance(image, label)
        else:
            # 验证集和测试集不进行数据增强
            pass
        
        """
        # 随机裁剪
        h, w = image.shape[:2]
        if h > self.patch_size and w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            image = image[top:top+self.patch_size, left:left+self.patch_size]
            label = label[top:top+self.patch_size, left:left+self.patch_size]
        else:
            # 如果图像小于patch_size，则进行填充
            image = np.pad(image, ((0, max(0, self.patch_size-h)), (0, max(0, self.patch_size-w)), (0,0)), mode='reflect')
            label = np.pad(label, ((0, max(0, self.patch_size-h)), (0, max(0, self.patch_size-w))), mode='reflect')
        """
        
        # 转换为tensor
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
            
        return image, label
    def get_img_name(self, idx):
        return os.path.splitext(os.path.basename(self.image_files[idx]))[0]
    def get_label_name(self, idx):
        return os.path.splitext(os.path.basename(self.label_files[idx]))[0]

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    from torchvision import transforms

    """
    # 创建数据集，使用数据增强(验证BuildingDataset类是否正确)
    val_dataset = BuildingDataset(
        image_dir=config.paths.val_data_dir,
        label_dir=config.paths.val_label_dir,
        mode="val",
    )
    print("数据集大小：", len(val_dataset))

    
    fig, axes = plt.subplots(10, 14, figsize=(60, 30))
    for i, (image, label) in enumerate(val_dataset):
        print("第{}对样本：".format(i+1))
        print("图片大小：", image.shape)
        print("标签大小：", label.shape)

        img_np = image.numpy().transpose(1, 2, 0)
        label_np = label.numpy()

        row = i // 7   # 每行显示7对样本   
        col = (i % 7) * 2  #每对样本占据2列

        # 显示图像
        axes[row, col].imshow(img_np)
        axes[row, col].axis("off")
        # 显示对应标签
        axes[row, col + 1].imshow(label_np, cmap='gray')
        axes[row, col + 1].axis("off")
    
    plt.tight_layout()
    plt.savefig("tmp/combined_val_dataset.png", dpi=300)
    plt.close()
    """
    

    
    # 创建数据集，使用数据增强(验证DataLoader是否正确)
    val_dataset = BuildingDataset(
        image_dir=config.paths.val_data_dir,
        label_dir=config.paths.val_label_dir,
        mode="train",
    )
    print("数据集大小：", len(val_dataset))

    # 创建数据加载器
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.hyperparams.batch_size, 
        shuffle=False,
        num_workers=config.hyperparams.num_workers,
    )

    for epoch in range(3):

        fig1, axes1 = plt.subplots(10, 14, figsize=(60, 30))
        fig2, axes2 = plt.subplots(10, 7, figsize=(40, 30))

        for i, (image, label) in enumerate(val_loader):
            print("------------第{}个epoch，第{}批次-----------".format(epoch+1, i+1))
            print("图片大小：", image.shape)
            print("标签大小：", label.shape)

            

            for idx in range(image.shape[0]):
                j = i * config.hyperparams.batch_size + idx

                # 获取单个样本
                image_single = image[idx].numpy().transpose(1, 2, 0)
                label_single = label[idx].numpy()

                # 计算正确的子图位置
                row = j // 7   # 每行显示7对样本   
                col = (j % 7) * 2  #每对样本占据2列

                # 显示图像
                axes1[row, col].imshow(image_single)
                axes1[row, col].axis("off")
                # 显示对应标签
                axes1[row, col + 1].imshow(label_single, cmap='gray')
                axes1[row, col + 1].axis("off")

                # 显示单独标签图表
                axes2.ravel()[j].imshow(label_single, cmap='gray')
        
        # 保存合并后的图像
        plt.figure(fig1.number)
        plt.tight_layout()
        plt.savefig("tmp/combined_val_dataset_{}.png".format(epoch+1), dpi=300)
        plt.close(fig1)

        # 保存单独标签图表
        plt.figure(fig2.number)
        plt.tight_layout()
        plt.savefig("tmp/label_only_{}.png".format(epoch+1), dpi=300)
        plt.close(fig2)
        

    
    

    