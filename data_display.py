import rasterio
import numpy as np
import os
import warnings
from rasterio.errors import NotGeoreferencedWarning
import matplotlib.pyplot as plt
from config_loader import load_config


def data_display(input_path, output_path, mode="train"):
    """
    显示数据集类别比例和数目
    :param input_path: 输入路径
    :param output_path: 输出路径
    :param mode: 模式，train, val, or test
    :return: None
    """
    
    # 抑制NotGeoreferencedWarning警告
    warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

    building_num = 0
    background_num = 0

    for filename in os.listdir(input_path):
        if filename.endswith(".tif"):
            label_path = os.path.join(input_path, filename)
            with rasterio.open(label_path) as src:
                label = src.read(1)
                label = (label == 255).astype(np.int64)

            building_num += np.sum(label == 1)
            background_num += np.sum(label == 0)

    #用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['KaiTi','SimHei','FangSong'] #汉字字体，优先使用楷体，其次是黑体，最后是仿宋
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    classes = ['建筑', '背景']
    values = [building_num, background_num]

    if mode == "train":
        name = "训练集"
    elif mode == "val":
        name = "验证集"
    elif mode == "test":
        name = "测试集"
    else:
        raise ValueError("mode should be train, val, or test")

    #用条形图来明确表示数量关系
    save_name = name + "-地物类别数目条形图"
    save_path = os.path.join(output_path, save_name + ".png")
    plt.figure(figsize=(10, 6))
    plt.barh(classes, values, color=['red', 'blue'])
    title = name + "-地物类别数目"
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    #用饼状图来明确表示比例关系
    save_name = name + "-地物类别比例饼状图"
    save_path = os.path.join(output_path, save_name + ".png")
    plt.figure(figsize=(8, 8))
    plt.pie(values, explode=[0.1,0], labels=classes, autopct='%1.1f%%',shadow=True,startangle=90,colors=['red', 'blue'])
    title = name + "-地物类别比例"
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':

    config = load_config()

    # 创建一级输出目录(.\data_display)
    output_path = config.paths.data_display_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # 创建二级输出目录(.data_display\raw_data、.data_display\data)
    output_path_rawdata = os.path.join(output_path, 'raw_data')
    if not os.path.exists(output_path_rawdata):
        os.makedirs(output_path_rawdata)
    output_path_data = os.path.join(output_path, 'data')
    if not os.path.exists(output_path_data):
        os.makedirs(output_path_data)


    # 显示原始数据
    train_label_path = config.paths.train_label_dir
    train_label_path = train_label_path.replace("data", "raw_data")
    data_display(train_label_path, output_path_rawdata, mode="train")

    val_label_path = config.paths.val_label_dir
    val_label_path = val_label_path.replace("data", "raw_data")
    data_display(val_label_path, output_path_rawdata, mode="val")

    test_label_path = config.paths.test_label_dir
    test_label_path = test_label_path.replace("data", "raw_data")
    data_display(test_label_path, output_path_rawdata, mode="test")

    # 显示整理后的数据
    train_label_path = config.paths.train_label_dir
    data_display(train_label_path, output_path_data, mode="train")

    val_label_path = config.paths.val_label_dir
    data_display(val_label_path, output_path_data, mode="val")

    test_label_path = config.paths.test_label_dir
    data_display(test_label_path, output_path_data, mode="test")
    
