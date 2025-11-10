import numpy as np
from pathlib import Path
from ultralytics import YOLO 
import json
import yaml
import argparse
with open("./config.json", "r") as cf:
    config = json.load(cf)

def validate_model(weights_path: Path, data_yaml_path: Path, subdataset: str) -> None:
    """Evaluate the provided weights on the test split and report aggregate metrics."""

    print(f"Validating weights: {weights_path}")
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

    parser = argparse.ArgumentParser(description="Validate the model on dataset.")
    validate_dataset_name=config["validate"]["validate_dataset"]

    parser.add_argument(
        "--dataset", 
        type=str, 
        default=validate_dataset_name,
        help="Dataset name for validation.")
    
    args = parser.parse_args()

    validate_cfg=config["validate"]["validate_"+args.dataset]
    subdataset=validate_cfg["subdataset"]
    validate_model_path=validate_cfg["validate_model_path"]
    data_yaml_path=validate_cfg["data_yaml_path"]

    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(validate_model_path),
        help="Path to the model weights file.",
    )
    parser.add_argument(
        "--YamlPath",
        type=Path,
        default=Path(data_yaml_path),
        help="Path to the dataset YAML file.",
    )

    parser.add_argument(
        "--subset", 
        type=str, 
        default=subdataset,
        help="Dataset name for validation.")
    
    args=parser.parse_args()
    weights_path=args.weights
    data_yaml_path=args.YamlPath
    validate_model(weights_path=weights_path, data_yaml_path=data_yaml_path,subdataset=args.subset)