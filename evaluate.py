import torch
import torch.nn as nn
import numpy as np
import rasterio
import os
import warnings
from rasterio.errors import NotGeoreferencedWarning
from models.unet import UNet
from models.unet_plus_plus import unet_plus_plus
from torch.utils.data import Dataset, DataLoader
from data_process import BuildingDataset
from config_loader import load_config
from get_metrics import get_confusion_matrix, get_confusion_matrix_old
from get_metrics import get_overall_accuracy, get_precision, get_recall, get_f1_score, get_iou_score, get_miou_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime,timedelta
from ruamel.yaml import YAML

def plot_confusion_matrix(confusion_matrix, class_names, save_path, xlabel='Predicted Label', ylabel='True Label', title='Confusion Matrix'):
    """
    绘制混淆矩阵
    :param confusion_matrix: 混淆矩阵
    :param class_names: 类别名称列表
    :param save_path: 保存路径  
    :param title: 图片标题
    """

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

    # 绘制混淆矩阵
    plt.figure(figsize = (14, 12))
    ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names,cbar_kws={"shrink":0.8, "aspect": 15})

    """
    # 设置colorbar刻度均匀分布
    cbar = ax.collections[0].colorbar
    max_val = confusion_matrix.max()
    cbar.set_ticks(np.linspace(0, max_val, 10))  # 设置10个均匀分布的刻度值
    cbar.set_ticklabels([f"{int(x)}" for x in np.linspace(0, max_val, 10)])  # 设置刻度值标签为整数
    """

    cbar = ax.collections[0].colorbar
    max_val = confusion_matrix.max()
    step = 1000000 if max_val >= 1000000 else (100000 if max_val >= 100000 else 50000)
    ticks = np.arange(0, max_val + step, step)
    cbar.set_ticks(ticks) 
    cbar.set_ticklabels([f"{int(x):,}" for x in ticks])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # 自动旋转标签
    #plt.xticks(rotation=45, ha='right') # 横坐标标签旋转45度，对齐方式为右对齐
    plt.xticks(rotation=0) # 横坐标标签不旋转
    plt.yticks(rotation=0) # 纵坐标标签不旋转

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_batch_sample_data(images, labels, predictions, config, img_names, label_names, save_dir):
    """
    可视化一个批次的样本数据（包括原始图像、标签和预测结果）
    :param images: 图像数据, 形状为(N, C, H, W)
    :param labels: 标签数据, 形状为(N, H, W)
    :param predictions: 预测结果数据, 形状为(N, H, W)
    :param img_names: 图像名称列表
    :param label_names: 标签名称列表
    :param save_dir: 保存路径
    """

    images = images.cpu().numpy().transpose(0,2,3,1) # NCHW -> NHWC
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()

    # 可视化每个样本
    for i in range(images.shape[0]):
        plt.figure(figsize=(18, 6))

        # 原始图像
        plt.subplot(1, 3, 1)
        img = (images[i] * 255.0).astype(np.uint8)
        plt.imshow(img)
        plt.title(f'Input Image\n{img_names[i]}')
        plt.axis('off')

        # 真实标签
        plt.subplot(1, 3, 2)
        plt.imshow(labels[i], cmap="gray", vmin=0, vmax=config.hyperparams.num_classes - 1)
        plt.title(f'Ground Truth\n{label_names[i]}')
        plt.axis("off")

        # 预测结果
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i], cmap="gray", vmin=0, vmax=config.hyperparams.num_classes - 1)
        plt.title(f'Prediction\n{img_names[i]}')
        plt.axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{img_names[i]}_result_visualization.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def save_metrics(metrics, save_path):
    """
    保存评估指标到txt文件中
    :param metrics: 评估指标字典
    :param save_path: 输出目录
    """
    with open(save_path, mode='w', encoding='utf-8') as f:

        # 写入总体指标
        f.write("Evaluation Metrics:\n")
        f.write(f"Mean IoU: {metrics['mIoU']:.4f}\n")
        f.write(f"Overall Accuracy: {metrics['OA']:.4f}\n")

        # 写入分类别指标表格
        f.write("Per-Class Metrics:\n")
        f.write("Class\tIoU\tPrecision\tRecall\tF1-Score\n")
        for i in range(len(metrics['IoU'])):
            line = f"{i}\t{metrics['IoU'][i]:.4f}\t{metrics['Precision'][i]:.4f}\t{metrics['Recall'][i]:.4f}\t{metrics['F1'][i]:.4f}\n"
            f.write(line)

def evaluate():

    # 加载配置
    config = load_config()

    # 设置随机种子
    seed_value = config.hyperparams.random_seed
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建一级输出目录（.\evaluate_results)
    output_dir = config.paths.evaluation_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 创建二级输出目录（.\evaluate_results\2025-05-15_12-00-00_evaluating)
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_dir, time_now + "_evaluating")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 创建三级输出目录（.\evaluate_results\2025-05-15_12-00-00_evaluating\metrics、.\evaluate_results\2025-05-15_12-00-00_evaluating\visualizations)
    output_dir_of_metrics = os.path.join(output_dir, "metrics")
    if not os.path.exists(output_dir_of_metrics):
        os.makedirs(output_dir_of_metrics)  
    output_dir_of_visualizations = os.path.join(output_dir, "visualizations")
    if not os.path.exists(output_dir_of_visualizations):
        os.makedirs(output_dir_of_visualizations)
        
    # 加载模型
    checkpoint_path = config.paths.model_load_path
    model = unet_plus_plus(encoder_name="resnet34",encoder_weights="imagenet",in_channels=config.data.channels,classes=config.hyperparams.num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # 创建测试数据集
    test_dataset = BuildingDataset(
        image_dir=config.paths.test_data_dir,
        label_dir=config.paths.test_label_dir,
        mode = "test",
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.hyperparams.batch_size, 
        shuffle=False, 
        num_workers=config.hyperparams.num_workers,
        pin_memory=config.hyperparams.pin_memory,
        prefetch_factor=config.hyperparams.prefetch_factor, 
    )


    # 初始化混淆矩阵
    confusion_mat = np.zeros((config.hyperparams.num_classes, config.hyperparams.num_classes))
    # 开始评估
    start_time = time.time() # 记录评估开始时间
    for batch_idx,(images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(images)
            predictions = torch.softmax(predictions, dim = 1)
            #print("predictions: ", torch.sum(predictions, dim = 1))
            predictions = torch.argmax(predictions, dim = 1)
            """
            #验证模型的输出是否已经通过sofmax函数进行了归一化处理(结果是模型的输出未经过softmax函数进行归一化处理)
            pre_sum = torch.sum(outputs, dim = 1)
            print("shape of pre_sum:", pre_sum.shape)
            print("pre_sum:", pre_sum)

            outputs = torch.softmax(outputs, dim = 1)
            print("shape of outputs after softmax:", outputs.shape)
            post_sum = torch.sum(outputs, dim = 1)
            print("shape of post_sum:", post_sum.shape) 
            print("post_sum:", post_sum)
            """

        if batch_idx == 0:
            # 仅对第一个batch进行可视化验证
            batch_img_filenames = [test_dataset.get_img_name(i) for i in range(batch_idx*config.hyperparams.batch_size, (batch_idx+1)*config.hyperparams.batch_size)]

            batch_label_filenames = [test_dataset.get_label_name(i) for i in range(batch_idx*config.hyperparams.batch_size, (batch_idx+1)*config.hyperparams.batch_size)]


            plot_batch_sample_data(images, labels, predictions, config, batch_img_filenames, batch_label_filenames, output_dir_of_visualizations)
        
        # 计算混淆矩阵
        confusion_mat += get_confusion_matrix(labels, predictions, config.hyperparams.num_classes)
        #confusion_mat += get_confusion_matrix_old(labels, predictions, config.hyperparams.num_classes)
        # 验证结果一致性
        #assert np.allclose(confusion_mat_old, confusion_mat), "结果不一致！"
        #print("结果一致！")
        #print("confusion_mat:", confusion_mat)

        """
        # 验证混淆矩阵计算是否正确
        true_class_counts = np.bincount(labels_np.flatten(), minlength=config.hyperparams.num_classes)
        pred_class_counts = np.bincount(outputs_np.flatten(), minlength=config.hyperparams.num_classes)
        print("\n类别分布验证：")
        print(f"{'类别':<8} {'真实样本数':<12} {'预测样本数':<12} {'差异率':<10}")
        for cls in range(config.hyperparams.num_classes):
            true_count = true_class_counts[cls]
            pred_count = pred_class_counts[cls]
            diff_ratio = abs(true_count - pred_count) / (true_count + 1e-8)
            print(f"{cls:<8} {true_count:<12} {pred_count:<12} {diff_ratio:.2%}")
        """

    # 绘制混淆矩阵
    class_names = list(config.data.class_mapping.values()) if hasattr(config.data, "class_mapping") else [str(i) for i in range(config.hyperparams.num_classes)]   # 类别名称列表

    confusion_matrix_filename ="Confusion-Matrix-(UNetPlusPlus-ResNet34)_" + "Input-Size({}x{}x{}x{})-NCHW_".format(config.hyperparams.batch_size,config.data.channels,config.data.patch_size,config.data.patch_size) + "Epoch({})_".format(config.hyperparams.num_epochs) + "Init-LR({})_".format(config.hyperparams.learning_rate) + ".png"
    confusion_matrix_save_path = os.path.join(output_dir_of_metrics, confusion_matrix_filename)

    plot_confusion_matrix(confusion_mat, class_names, confusion_matrix_save_path, xlabel='预测类别', ylabel='真实类别', title='混淆矩阵')
    

    # 计算评估指标
    OA = get_overall_accuracy(confusion_mat)
    #print(f"OA: {OA:.4f}")
    precision = get_precision(confusion_mat)
    #print(f"Precision: {precision}")
    recall = get_recall(confusion_mat)
    #print(f"Recall: {recall}")
    f1_score = get_f1_score(confusion_mat)  
    #print(f"F1-score: {f1_score}")
    iou_score = get_iou_score(confusion_mat)
    #print(f"IoU-score: {iou_score}")
    miou_score = get_miou_score(confusion_mat)
    #print(f"mIoU-score: {miou_score}")
    metrics ={'OA': OA,'mIoU': miou_score, 'IoU': iou_score, 'Precision': precision, 'Recall': recall, 'F1': f1_score}

    metrics_filename = "Metrics-Report-(UNetPlusPlus-ResNet34)_" + "Input-Size({}x{}x{}x{})-NCHW_".format(config.hyperparams.batch_size,config.data.channels,config.data.patch_size,config.data.patch_size) + "Epoch({})_".format(config.hyperparams.num_epochs) + "Init-LR({})_".format(config.hyperparams.learning_rate) + ".txt"
    metrics_save_path = os.path.join(output_dir_of_metrics, metrics_filename)

    save_metrics(metrics, metrics_save_path)

    # 保存此次评估所使用的配置文件
    save_config_name = "Evaluating-Config(UNetPlusPlus-ResNet34)_" + "Input-Size({}x{}x{}x{})-NCHW_".format(config.hyperparams.batch_size,config.data.channels,config.data.patch_size,config.data.patch_size) + "Epoch({})_".format(config.hyperparams.num_epochs) + "Init-LR({})_".format(config.hyperparams.learning_rate) + ".yml"
    save_config_path = os.path.join(output_dir, save_config_name)
    yaml = YAML()
    yaml.preserve_quotes = True 
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(save_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f)


    end_time = time.time() # 记录评估结束时间
    elapsed_time = timedelta(seconds=int(end_time - start_time)) # 计算评估用时
    print("Evaluation completed in:", elapsed_time)

        
if __name__ == "__main__":
    evaluate()

