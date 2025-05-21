import os
import rasterio
import shutil
import warnings
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

def data_clean(data_dir, label_dir):
    """
    清理数据集，删除无效文件（主要是删除非tiff/tif格式的文件，例如使用envi和arcgis软件打开之后保存的金字塔文件）
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


def crop_data(input_img_dir, input_label_dir, output_img_dir, output_label_dir, crop_size, stride):
    """
    裁剪原始数据
    :param input_img_dir: 原始图像文件夹路径
    :param input_label_dir: 原始标签文件夹路径
    :param output_img_dir: 裁剪后的图像文件夹路径
    :param output_label_dir: 裁剪后的标签文件夹路径
    :param crop_size: 裁剪大小
    :param stride: 裁剪步长
    """

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    # 抑制NotGeoreferencedWarning警告
    warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

    os.environ['PROJ_LIB'] = r"G:\Anaconda3\Lib\site-packages\pyproj\proj_dir\share\proj"  # 替换为本地的proj库路径
    
    data_clean(input_img_dir, input_label_dir)  # 清理无效文件

    for filename in os.listdir(input_img_dir):
        label_filename = filename.replace(".tiff", ".tif")
        img_path = os.path.join(input_img_dir, filename)    # 原始图像路径
        label_path = os.path.join(input_label_dir, label_filename)  # 原始标签路径

        with rasterio.open(img_path) as src_img:
            with rasterio.open(label_path) as src_label:
                width = src_img.width
                height = src_img.height
                tf = src_img.transform  # 影像的仿射变换矩阵
                crs = src_img.crs  # 影像的坐标系信息
                
                for y in range(0, height - crop_size + 1, stride):
                    for x in range(0, width - crop_size + 1, stride):
                        window = Window(x, y, crop_size, crop_size)
                        img_crop = src_img.read(window=window)
                        label_crop = src_label.read(window=window)

                        window_transform = rasterio.windows.transform(window, tf)  # 计算裁剪窗口的仿射变换矩阵

                        # 如果影像中有建筑物并且不包含nodata空值，则保存为裁剪后的影像和label
                        #if label_crop.any() and not (img_crop == 255).any():
                        output_img_path = os.path.join(output_img_dir, f"{filename.split('.')[0]}_y{y:04d}_x{x:04d}.tiff")
                        output_label_path = os.path.join(output_label_dir, f"{label_filename.split('.')[0]}_y{y:04d}_x{x:04d}.tif")

                        profile = src_img.profile
                        profile['width'], profile['height'], profile['transform'],profile['crs'] = crop_size, crop_size, window_transform, crs
                        with rasterio.open(output_img_path, 'w', **profile) as dst_img:
                            dst_img.write(img_crop)

                        profile = src_label.profile
                        profile['width'], profile['height'], profile['transform'],profile['crs'] = crop_size, crop_size, window_transform, crs
                        with rasterio.open(output_label_path, 'w', **profile) as dst_label:
                            dst_label.write(label_crop)

def data_manage(raw_data_dir,output_dir, crop_size, stride):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理训练集
    raw_train_dir = os.path.join(raw_data_dir, "train")
    raw_train_label_dir = os.path.join(raw_data_dir, "train_labels")
    output_train_dir = os.path.join(output_dir, "train")
    output_train_label_dir = os.path.join(output_dir, "train_labels")

    data_clean(raw_train_dir, raw_train_label_dir)  # 清理无效文件
    crop_data(raw_train_dir, raw_train_label_dir, output_train_dir, output_train_label_dir, crop_size, stride)

    # 处理验证集
    raw_val_dir = os.path.join(raw_data_dir, "val")
    raw_val_label_dir = os.path.join(raw_data_dir, "val_labels")
    output_val_dir = os.path.join(output_dir, "val")
    output_val_label_dir = os.path.join(output_dir, "val_labels")

    data_clean(raw_val_dir, raw_val_label_dir)  # 清理无效文件
    crop_data(raw_val_dir, raw_val_label_dir, output_val_dir, output_val_label_dir, crop_size, stride)

    # 处理测试集
    raw_test_dir = os.path.join(raw_data_dir, "test")
    raw_test_label_dir = os.path.join(raw_data_dir, "test_labels")
    output_test_dir = os.path.join(output_dir, "test")
    output_test_label_dir = os.path.join(output_dir, "test_labels")

    data_clean(raw_test_dir, raw_test_label_dir)  # 清理无效文件
    crop_data(raw_test_dir, raw_test_label_dir, output_test_dir, output_test_label_dir, crop_size, stride)

if __name__ == "__main__":

    raw_data_dir = r".\raw_data"  # 原始数据文件夹路径
    output_dir = r".\data"  # 输出数据文件夹路径
    crop_size = 256  # 裁剪大小
    stride = 256  # 裁剪步长
    data_manage(raw_data_dir, output_dir, crop_size, stride)
