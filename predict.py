import rasterio
import warnings
import numpy as np
import os
import math
from rasterio.errors import NotGeoreferencedWarning
from config_loader import load_config
from models.unet_plus_plus import unet_plus_plus
import torch
import time
from datetime import datetime, timedelta

def load_model(model_path, in_channels, classes, device):
    """加载训练好的模型"""
    #model = UNet().to(device)
    model = unet_plus_plus(encoder_name="resnet34",encoder_weights="imagenet",in_channels=in_channels,classes=classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def readTif(img_path):
    """读取tif影像
    :param img_path: 影像路径
    :return: 影像数据，仿射变换矩阵，地理坐标系信息
    """
    
    os.environ['PROJ_LIB'] = r"G:\Anaconda3\Lib\site-packages\pyproj\proj_dir\share\proj"  

    with rasterio.open(img_path) as src:
        warnings.simplefilter('ignore', NotGeoreferencedWarning)  # 忽略NotGeoreferencedWarning警告
        # 读取所有波段
        image = src.read()
        image = image.astype(np.float32) / 255.0  # 归一化到[0, 1]
        # 获取图像的仿射变换矩阵和地理坐标系信息
        tf = src.transform  # 仿射变换矩阵
        crs = src.crs  # 地理坐标系信息

    return image, tf, crs

def writeTiff(im_data, im_geotrans, im_proj, path):

    os.environ['PROJ_LIB'] = r"G:\Anaconda3\Lib\site-packages\pyproj\proj_dir\share\proj"

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = im_data[np.newaxis, :, :]
        im_bands, im_height, im_width = im_data.shape
    with rasterio.open(path, "w", driver="GTiff", height=im_height, width=im_width, count=im_bands, dtype=im_data.dtype, crs=im_proj, transform=im_geotrans) as dst:
        dst.write(im_data)

#  tif裁剪（tif像素数据，裁剪边长）
def TifCroppingArray(img, SideLength):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (256 - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (256 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (256 - SideLength * 2): i * (256 - SideLength * 2) + 256,
                      j * (256 - SideLength * 2): j * (256 - SideLength * 2) + 256]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (256 - SideLength * 2): i * (256 - SideLength * 2) + 256,
                  (img.shape[1] - 256): img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 256): img.shape[0],
                  j * (256 - SideLength * 2): j * (256 - SideLength * 2) + 256]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - 256): img.shape[0],
              (img.shape[1] - 256): img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (256 - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (256 - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver

def results_combine(shape, TifArray, predicts, RepetitiveLength, RowOver, ColumnOver):
    """
    将裁剪后的图像块的预测结果拼接成最终的预测结果
    :param shape: 图像的形状
    :param TifArray: 裁剪后的图像块
    :param predicts: 预测结果列表
    :param RepetitiveLength: 重叠区域长度
    :param RowOver: 行上的剩余数
    :param ColumnOver: 列上的剩余数
    :return: 最终的预测结果
    """
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0
    for i, img in enumerate(predicts):
        #print("shape of img: ", img.shape)
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if (i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                #print("shape of result: ", result[0: 256 - RepetitiveLength, 0: 256 - RepetitiveLength].shape)
                #print("shape of img[0: 256 - RepetitiveLength, 0: 256 - RepetitiveLength]: ", img[0: 256 - RepetitiveLength, 0: 256 - RepetitiveLength].shape)
                result[0: 256 - RepetitiveLength, 0: 256 - RepetitiveLength] = img[0: 256 - RepetitiveLength, 0: 256 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                #  原来错误的
                # result[shape[0] - ColumnOver : shape[0], 0 : 256 - RepetitiveLength] = img[0 : ColumnOver, 0 : 256 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0: 256 - RepetitiveLength] = img[256 - ColumnOver - RepetitiveLength: 256, 0: 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                0:256 - RepetitiveLength] = img[RepetitiveLength: 256 - RepetitiveLength, 0: 256 - RepetitiveLength]
                #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif (i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 256 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0: 256 - RepetitiveLength, 256 - RowOver: 256]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0], shape[1] - RowOver: shape[1]] = img[256 - ColumnOver: 256, 256 - RowOver: 256]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                shape[1] - RowOver: shape[1]] = img[RepetitiveLength: 256 - RepetitiveLength, 256 - RowOver: 256]
                #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 256 - RepetitiveLength,(i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength] = img[0: 256 - RepetitiveLength, RepetitiveLength: 256 - RepetitiveLength]
                #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0],(i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength] = img[256 - ColumnOver: 256, RepetitiveLength: 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) +RepetitiveLength,] = img[RepetitiveLength: 256 - RepetitiveLength, RepetitiveLength: 256 - RepetitiveLength]
    return result


def predict_small_image(model, image_path, device, color_map, save_path=None):
    """
    对单张小尺寸图像进行预测（例如尺寸为256x256的图像）
    :param model: 加载的模型
    :param image_path: 图像路径
    :param device: 设备
    :param color_map: 颜色映射字典
    :param save_path: 预测结果保存路径
    :return: 预测结果
    """

    # 读取图像
    image, tf, crs = readTif(image_path)
    channels, height, width = image.shape  # 获取图像的通道数、高度、宽度
    image = torch.from_numpy(image)
    image = image.unsqueeze(0).to(device)  # 增加batch维度, 形状为1×C×H×W

    # 进行预测
    with torch.no_grad():
        output = model(image) 
        output = torch.softmax(output, dim=1)  # 应用softmax
        output = torch.argmax(output, dim=1)  # 取最大值索引作为预测结果
        output = output.squeeze(0).cpu().numpy().astype(np.uint8)  # 去掉batch维度, 形状为H×W
    
    # 保存预测结果
    result = np.zeros((channels, height, width), dtype=np.uint8)  # 创建保存结果的数组
    color_map = {k: [int(x) for x in v.split(",")] for k, v in color_map.items()} # 将配置文件的颜色映射转换为字典
    for cls, color in color_map.items():
        for i in range(channels):  # 遍历所有波段
            result[i][output == cls] = color[i]  

    if save_path:
        writeTiff(result, tf, crs, save_path)  # 写入预测结果
    
    return result

def predict_large_image(model, image_path, patch_size,center_area_perc, device, save_path=None):
    """
    对单张大尺寸图像进行预测（例如尺寸为10000x10000的图像）。
    在模型预测过程中，如果将较大的待分类遥感影像直接输入到网络模型中会造成内存溢出，故一般将待分类图像裁剪为一系列较小图像批量输入到模型中进行预测，然后再将预测结果按照裁剪顺序拼接成一张最终的结果图像。裁剪过程中，为了避免裁剪出的图像边缘信息损失，通常会采用重叠裁剪的方式，即将图像按照一定的步长进行裁剪，每次裁剪出的图像大小为crop_size，且相邻裁剪出的图像之间有一定的重叠区域，重叠区域大小为overlap。这样做的好处是可以保留图像的边缘信息，避免边缘信息的丢失，同时也可以减少模型的输入数据量，从而提高模型的预测效率。
    :param model: 加载的模型
    :param image_path: 图像路径
    :param patch_size: 裁剪尺寸
    :param center_area_perc: 图像块拼接时保留的中心区域比例，范围为0到1之间，例如0.8表示保留中心区域的80%。
    :param device: 设备
    :return: 预测结果（单通道）
    """
    
    RepetitiveLength = int((1 - math.sqrt(center_area_perc)) * patch_size / 2)  # 计算重叠区域长度
    #  读取图像
    image, tf, crs = readTif(image_path)
    channels, height, width = image.shape  # 获取图像的通道数、高度、宽度
    image = image.swapaxes(2, 0).swapaxes(1, 0) # 此时，图像形状调整为H×W×C

    TifArray, RowOver, ColumnOver = TifCroppingArray(image, RepetitiveLength)  # 裁剪图像

    predicts = []  # 用于存储预测结果的列表
    for i in range(len(TifArray)):
        for j in range(len(TifArray[0])):
            #  读取裁剪图像
            image = TifArray[i][j]
            image = image.transpose(2, 0, 1)  # 此时，图像形状调整为C×H×W
            image = torch.from_numpy(image)
            image = image.unsqueeze(0).to(device)  # 增加batch维度, 形状为1×C×H×W

            # 初始化预测结果
            predict = np.zeros(image.shape[1:])

            # 进行预测
            with torch.no_grad():
                output = model(image) 
                output = torch.softmax(output, dim=1)  # 应用softmax
                output = torch.argmax(output, dim=1)  # 取最大值索引作为预测结果
                output = torch.squeeze(output).cpu().numpy().astype(np.uint8)  # 去掉batch维度, 形状为H×W
                predict = output
            
            #  保存小图块的预测结果
            predicts.append(predict)
    
    #  拼接预测结果
    result_shape = (height,width)
    result = results_combine(result_shape, TifArray, predicts, RepetitiveLength, RowOver, ColumnOver)
    
    #  保存拼接之后的预测结果
    if save_path:
        writeTiff(result, tf, crs, save_path)  # 写入预测结果
    
    return result


if __name__ == '__main__':
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建一级输出目录(.\predict_results)
    output_dir = config.paths.prediction_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建二级输出目录(.\predict_results\2025-05-16_20-07-49_predicting)
    time_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(output_dir, time_now + "_predicting")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建三级输出目录（.\predict_results\2025-05-16_20-07-49_predicting\patches_predictions、.\predict_results\2025-05-16_20-07-49_predicting\whole_image_predictions）
    output_dir_of_patches = os.path.join(output_dir, "patches_predictions")
    if not os.path.exists(output_dir_of_patches):
        os.makedirs(output_dir_of_patches)
    output_dir_of_whole_image = os.path.join(output_dir, "whole_image_predictions")
    if not os.path.exists(output_dir_of_whole_image):
        os.makedirs(output_dir_of_whole_image)

    # 加载模型
    model_path = config.paths.model_load_path
    model = load_model(model_path, in_channels=config.data.channels, classes=config.hyperparams.num_classes, device=device)

    # 预测小尺寸图像
    start_time = time.time()
    patch_path = r"E:\buildings_extract\data\test\22828930_15_y1024_x0768.tiff"
    patch_name = os.path.basename(patch_path)
    save_path = os.path.join(output_dir_of_patches, patch_name.split(".")[0] + "_predict.tif")
    predict_small_image(model, patch_path, device, config.data.color_mapping, save_path)
    print("预测小尺寸图像用时：", timedelta(seconds=int(time.time() - start_time)))


    # 预测大尺寸图像
    start_time = time.time()
    large_image_path = r"E:\buildings_extract\raw_data\test\23729035_15.tiff"
    large_image_name = os.path.basename(large_image_path)
    save_path = os.path.join(output_dir_of_whole_image, large_image_name.split(".")[0] + "_predict.tif")
    predict_large_image(model, large_image_path, config.data.patch_size, config.data.center_area_perc, device, save_path)
    print("预测大尺寸图像用时：", timedelta(seconds=int(time.time() - start_time)))

