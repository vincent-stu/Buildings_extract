import os
import rasterio
from rasterio import features
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from config_loader import load_config

def raster_to_vector(raster_path, vector_path,field_name="class", ignore_value=None, class_map=None,color_map=None):
    """
    将语义分割得到的栅格文件（一般为单波段分类结果）转换为矢量文件，并将栅格值作为字段添加到矢量文件中。
    :param raster_path: 栅格文件路径
    :param vector_path: 矢量文件路径
    :param field_name: 字段名称
    :param ignore_value: 忽略值
    :param class_map: 类别映射字典，格式为{value: label}，其中value为栅格值，1为建筑，0为背景。如果是单波段的分类结果，不需要传入该参数。
    :param color_map: 颜色映射字典，格式为{value: (R, G, B)}，其中value为类别值，1为建筑，0为背景。如果是单波段的分类结果，不需要传入该参数。
    :return:
    """

    os.environ['PROJ_LIB'] = r"G:\Anaconda3\Lib\site-packages\pyproj\proj_dir\share\proj" 

    # 读取栅格文件
    with rasterio.open(raster_path) as src:

        channels = src.count
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

        if channels == 1:  # 单波段分类结果
            # 创建矢量文件
            vector = (
                {"properties":{field_name: v}, "geometry": s} for i, (s, v) in enumerate(
                    features.shapes(src.read(1), transform=transform, connectivity=8)
            )
            if v!= ignore_value   # 过滤忽略值
            )

            # 创建GeoDataFrame并保存
            gdf = gpd.GeoDataFrame.from_features(vector, crs = crs.to_wkt())
            gdf.to_file(vector_path, driver='ESRI Shapefile')
        else:  # 三波段分类结果
            pred_data = src.read().transpose(1,2,0)  # 转换为(height, width, channels)的形状
            class_index = np.zeros(pred_data.shape[:2], dtype=np.uint8)  # 创建一个与图像大小相同的数组，用于存储类别索引
            color_map = {k:[int(x) for x in v.split(",")] for k, v in color_map.items()} #将配置文件中的颜色映射转换为字典
            for cls_id, (r, g, b) in color_map.items():
                class_index[np.all(pred_data == [r, g, b], axis=2)] = cls_id     # 找到每个像素对应的类别索引

            # 创建矢量文件
            vector = (
                {"properties":{field_name:v}, "geometry": s} for i, (s, v) in enumerate(
                    features.shapes(class_index, transform=transform, connectivity=8)
                )
                if v!= ignore_value   # 过滤忽略值
            )

            # 创建GeoDataFrame并保存
            gdf = gpd.GeoDataFrame.from_features(vector, crs = crs.to_wkt())
            gdf.to_file(vector_path, driver='ESRI Shapefile')

if __name__ == "__main__":

    config = load_config()

    #单波段预测结果转换为矢量
    raster_path = r"E:\buildings_extract\predict_results\2025-05-17_21-16-20_predicting\whole_image_predictions\23729035_15_predict.tif"
    vector_path = r"E:\buildings_extract\predict_results\2025-05-17_21-16-20_predicting\whole_image_predictions\23729035_15_whole_image_predictions.shp"
    raster_to_vector(raster_path, vector_path)

    #三波段预测结果转换为矢量
    raster_path = r"E:\buildings_extract\predict_results\2025-05-17_21-16-20_predicting\patches_predictions\22828930_15_y1024_x0768_predict.tif"
    vector_path = r"E:\buildings_extract\predict_results\2025-05-17_21-16-20_predicting\patches_predictions\22828930_15_y1024_x0768_patches_predictions.shp"
    raster_to_vector(raster_path, vector_path, color_map=config.data.color_mapping)




