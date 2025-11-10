import rasterio

# 打开 TIFF 文件
with rasterio.open('dataset/AIR-SARShip-1.0/SARShip-1.0-1/SARShip-1.0-1.tiff') as dataset:
    # 获取仿射变换矩阵
    transform = dataset.transform
    print("Transform:", transform)
    
    # 获取地理坐标参考系统（CRS）
    crs = dataset.crs
    print("CRS:", crs)