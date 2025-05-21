import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from models.unet import UNet
from models.unet_plus_plus import unet_plus_plus
import matplotlib.pyplot as plt
import numpy as np
from data_process import BuildingDataset
import time
from datetime import timedelta
from config_loader import load_config
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss
from pytorch_toolbelt import losses as L
from datetime import datetime
from ruamel.yaml import YAML
import pandas as pd

def plot_loss_curve(train_loss_epoch, val_loss_epoch, save_path):
    """
    绘制损失曲线
    :param train_loss_epoch: 训练损失列表(list)
    :param val_loss_epoch: 验证损失列表(list)
    :param save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    plt.plot(train_loss_epoch, label='train_loss', color='blue',linewidth=2)
    plt.plot(val_loss_epoch, label='val_loss', color='red',linewidth=2)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    actual_epochs = len(train_loss_epoch) # 实际训练轮数
    tick_interval = max(actual_epochs // 20, 2)  # 间隔为实际训练轮数的20分之一
    plt.xticks(range(0, actual_epochs + 1, tick_interval))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper right", frameon=True, edgecolor="black", fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def train():

    # 加载配置
    config = load_config()

    # 设置随机种子
    seed_value = config.hyperparams.random_seed
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    # 创建一级输出目录(.\train_results)
    output_dir = config.paths.train_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 创建二级输出目录(.\train_results\2025-05-04_12-00-00_training)
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
    output_dir = os.path.join(output_dir, time_now + "_training")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 创建三级输出目录(.\train_results\2025-05-04_12-00-00_training\checkpoints、.\train_results\2025-05-04_12-00-00_training\visualizations)
    output_dir_of_checkpoints = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(output_dir_of_checkpoints):
        os.makedirs(output_dir_of_checkpoints)
    output_dir_of_visualizations = os.path.join(output_dir, "visualizations")
    if not os.path.exists(output_dir_of_visualizations):
        os.makedirs(output_dir_of_visualizations)
    
    # 创建数据集
    train_dataset = BuildingDataset(
        image_dir=config.paths.train_data_dir,
        label_dir=config.paths.train_label_dir,
        mode = "train",
    )
    
    val_dataset = BuildingDataset(
        image_dir=config.paths.val_data_dir,
        label_dir=config.paths.val_label_dir,
        mode = "val",
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.hyperparams.batch_size, 
        shuffle=True, 
        num_workers=config.hyperparams.num_workers,
        pin_memory=config.hyperparams.pin_memory,
        prefetch_factor=config.hyperparams.prefetch_factor
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.hyperparams.batch_size, 
        shuffle=False, 
        num_workers=config.hyperparams.num_workers,
        pin_memory=config.hyperparams.pin_memory,
        prefetch_factor=config.hyperparams.prefetch_factor
    )
    
    # 初始化模型
    #model = UNet().to(device)
    model = unet_plus_plus(encoder_name="resnet34",encoder_weights="imagenet",in_channels=config.data.channels,classes=config.hyperparams.num_classes).to(device)
    
    # 定义损失函数和优化器
    #criterion = nn.CrossEntropyLoss()
    DiceLoss_fn = DiceLoss(mode='multiclass').to(device)
    SoftCrossEntropyLoss_fn = SoftCrossEntropyLoss(smooth_factor=config.hyperparams.smooth_factor).to(device)
    #loss_fn = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropyLoss_fn,first_weight=0.5,second_weight=0.5).to(device)
    loss_fn = lambda x, y: config.hyperparams.diceloss_weight * DiceLoss_fn(x, y) + config.hyperparams.softcrossentropyloss_weight * SoftCrossEntropyLoss_fn(x, y)

    optimizer = optim.AdamW(model.parameters(), lr=config.hyperparams.learning_rate, weight_decay=config.hyperparams.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'min', 
        patience=config.hyperparams.lr_patience, 
        factor=config.hyperparams.lr_factor,
        min_lr=config.hyperparams.min_lr,
    )
    
    # 训练循环
    best_val_loss = float('inf')
    
    # 记录总训练时间
    total_start_time = time.time()
    # 记录训练过程中的训练损失和验证损失
    train_loss_epoch, val_loss_epoch = [], []
    # 添加早停计数器和耐心值
    early_stop_counter = 0
    early_stop_patience = config.hyperparams.early_stop_patience
    
    # 记录训练过程中的各项指标
    metircs = []

    for epoch in range(config.hyperparams.num_epochs):
        # 记录epoch开始时间
        epoch_start_time = time.time()
        
        model.train()
        running_loss = []
        
        # 记录训练阶段时间
        train_start_time = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())
        train_time = time.time() - train_start_time
        train_loss_epoch.append(np.array(running_loss).mean())


        # 验证
        model.eval() 
        val_loss = []
        
        # 记录验证阶段时间
        val_start_time = time.time()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).long()
                outputs = model(images)
                val_loss.append(loss_fn(outputs, labels).item())
        val_time = time.time() - val_start_time
        val_loss_epoch.append(np.array(val_loss).mean())
        # 更新学习率
        scheduler.step(np.array(val_loss).mean())

        # 计算epoch总耗时
        epoch_time = time.time() - epoch_start_time

        epoch_metrics = {"epoch": epoch + 1, "train_loss": np.array(running_loss).mean(), "val_loss": np.array(val_loss).mean(), "lr": optimizer.param_groups[0]["lr"], "train_time": train_time, "val_time": val_time, "epoch_time": epoch_time}
        metircs.append(epoch_metrics)
        
        print(f'Epoch [{epoch+1}/{config.hyperparams.num_epochs}], '
              f'Train Loss: {np.array(running_loss).mean():.4f}, '
              f'Val Loss: {np.array(val_loss).mean():.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
              f'Train Time: {timedelta(seconds=int(train_time))}, '
              f'Val Time: {timedelta(seconds=int(val_time))}, '
              f'Epoch Time: {timedelta(seconds=int(epoch_time))}')

        # 保存最佳模型
        if np.array(val_loss).mean() < best_val_loss:

            early_stop_counter = 0  # 重置早停计数器
            best_val_loss = np.array(val_loss).mean()

            # 明确模型输出名字和路径
            checkpoins_name = "Building-Extract-Model(UNetPlusPlus-ResNet34)_" + "Input-Size({}x{}x{}x{})-NCHW_".format(config.hyperparams.batch_size,config.data.channels,config.data.patch_size,config.data.patch_size) + "Epoch({})_".format(config.hyperparams.num_epochs) + "Init-LR({})_".format(config.hyperparams.learning_rate) + "_best-weights.pth"
            model_path = os.path.join(output_dir_of_checkpoints, checkpoins_name)

            # 保存模型参数
            torch.save(model.state_dict(), model_path)

            # 对config.yml中的model_load_path进行更新
            config.paths.model_load_path = model_path
            yaml = YAML()
            yaml.preserve_quotes = True 
            yaml.indent(mapping=2, sequence=4, offset=2)
            with open("configs.yml", "w", encoding="utf-8") as f:
                yaml.dump(config.model_dump(), f)      

            #将配置文件保存到output_dir目录下，以便随时查看训练时的配置信息。
            save_config_name = "Training-Config(UNetPlusPlus-ResNet34)_" + "Input-Size({}x{}x{}x{})-NCHW_".format(config.hyperparams.batch_size,config.data.channels,config.data.patch_size,config.data.patch_size) + "Epoch({})_".format(config.hyperparams.num_epochs) + "Init-LR({})_".format(config.hyperparams.learning_rate) + ".yml"
            save_config_path = os.path.join(output_dir, save_config_name)
            with open(save_config_path, "w", encoding="utf-8") as f:
                yaml.dump(config.model_dump(), f)

        else:
            early_stop_counter += 1  # 计数器加一
            if early_stop_counter >= early_stop_patience:
                print(f'Early Stopping at Epoch {epoch+1}')
                break
        
    
    # 计算总训练时间
    total_time = time.time() - total_start_time
    print(f'\nTotal Training Time: {timedelta(seconds=int(total_time))}')

    # 保存训练过程中的各项指标
    metrics_df = pd.DataFrame(metircs)
    metrics_df["train_time_str"] = metrics_df["train_time"].apply(lambda x: str(timedelta(seconds=int(x))))
    metrics_df["val_time_str"] = metrics_df["val_time"].apply(lambda x: str(timedelta(seconds=int(x))))
    metrics_df["epoch_time_str"] = metrics_df["epoch_time"].apply(lambda x: str(timedelta(seconds=int(x))))
    excel_name = "Training-Metrics(UNetPlusPlus-ResNet34)_" + "Input-Size({}x{}x{}x{})-NCHW_".format(config.hyperparams.batch_size,config.data.channels,config.data.patch_size,config.data.patch_size) + "Epoch({})_".format(config.hyperparams.num_epochs) + "Init-LR({})_".format(config.hyperparams.learning_rate) + ".xlsx"
    excel_path = os.path.join(output_dir_of_visualizations, excel_name)
    metrics_df.to_excel(excel_path, index=False, engine='openpyxl')
    
    # 绘制训练过程中的损失曲线
    loss_curve_name = "Training-Loss-Curve(UNetPlusPlus-ResNet34)_" + "Input-Size({}x{}x{}x{})-NCHW_".format(config.hyperparams.batch_size,config.data.channels,config.data.patch_size,config.data.patch_size) + "Epoch({})_".format(config.hyperparams.num_epochs) + "Init-LR({})_".format(config.hyperparams.learning_rate) + ".png"
    loss_curve_path = os.path.join(output_dir_of_visualizations, loss_curve_name)
    plot_loss_curve(train_loss_epoch, val_loss_epoch, loss_curve_path)


if __name__ == '__main__':
    train() 