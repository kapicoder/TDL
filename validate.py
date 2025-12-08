import numpy as np
from pathlib import Path
from ultralytics import YOLO 
import json
import yaml
import argparse
from utils.config import CONFIG

def validate_model(cfg) -> None:
    """Evaluate the provided weights on the test split and report aggregate metrics."""
    weights_path = cfg["validate_model_path"]
    data_yaml_path = cfg["validate_data_yaml_path"]
    subdataset=cfg["validate_subdataset"]
    print(f"Validating weights: {weights_path}")
    print(f"Using dataset: {data_yaml_path}")
    print(f"Using subdataset: {subdataset}")
    model = YOLO(str(weights_path))
    res = model.val(data=data_yaml_path, split=subdataset)

    # 1) 取混淆矩阵：(nc+1, nc+1)，最后一行/列为 background
    cm = res.confusion_matrix.matrix
    cm_fg = cm[:-1, :-1]                 # 只看前 nc 类，忽略背景
    with open(data_yaml_path,"r") as f:
        data_dict = yaml.safe_load(f)
    nc= data_dict['nc']
    name=data_dict['names']
    acc=np.array(np.zeros(nc+1))
    far=np.array(np.zeros(nc+1))
    rec=np.array(np.zeros(nc+1))
    for i in range(nc):
        tp = np.diag(cm)[i]            # 对角 = TP
        fp = cm.sum(axis=1)[i] - np.diag(cm)[i]
        fn = cm.sum(axis=0)[i] - np.diag(cm)[i]
        acc[i] = tp / (tp + fp + 1e-9)          # 检测准确率
        far[i] = fp / (tp + fp + 1e-9)          # 虚警率
        rec[i] = tp / (tp + fn + 1e-9)          # 召回率
        print(f"类别 {name[i]} 的评估结果：")
        print(f"tp: {tp}, fp: {fp}, fn: {fn}")
        print(f"  准确率 (Precision) : {acc[i]:.4f}")
        print(f"  虚警率 (FAR)      : {far[i]:.4f}")
        print(f"  召回率 (Recall)   : {rec[i]:.4f}\n")
        #计算AP
    rec[i]=0
    acc[i]=1
    ap=0
    for i in range(nc):
        ap+=(rec[i]-rec[i+1])*acc[i]
    print(f"平均精确度ap:{ap}")
        # print(f"评估完成！ {metrics.box.map}")



if __name__== "__main__" :
    config=CONFIG()
    parser = argparse.ArgumentParser(description="Validate the model on dataset.")
    validate_dataset_name=config["validate_dataset"]

    parser.add_argument(
        "--dataset", 
        type=str, 
        default=validate_dataset_name,
        help="Dataset name for validation.")


    parser.add_argument(
        "--weights",
        type=Path,
        help="Path to the model weights file.",
    )
    parser.add_argument(
        "--YamlPath",
        type=Path,
        help="Path to the dataset YAML file.",
    )

    parser.add_argument(
        "--subset", 
        type=str, 
        help="Dataset name for validation.")
    args=parser.parse_args()
    config.update_config(**vars(args))
    validate_model(config)