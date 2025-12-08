对 SAR 图像进行识别和定位以及裁剪，并且返回经纬度坐标

dataset: 其中包含所有可能需要使用的数据集

pre_trained_model: 其中包含可能需要使用的 yolo11 的预训练模型

result: 其中包含训练或者测试的结果

runs: 其中包含使用 yolo11 验证数据的结果

target_detection_location: 使用 rgb 图像进行识别

visualization: 可视化各个数据集

utils: 工具函数存放

test_single: 对整张图像进行识别

batch_convert: 调用 test_single 对文件夹进行递归的识别

batch_split_api: 现在是对单张图像进行的 api 接口

batch_spilt_request: 对整张图像进行识别的 request
