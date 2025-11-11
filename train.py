from __future__ import annotations

import argparse
import numpy as np
import tempfile
from pathlib import Path
import yaml
from ultralytics import YOLO  # type: ignore
import json
with open("./config.json", "r") as cf:
    config = json.load(cf)

def train_model(
) -> None:
    """Train YOLO on MAR_reset using the provided pretrained weights."""
    
    dataset_name=config["train"]["train_dataset"]
    dataset_cfg=config["train"]["train_"+dataset_name]

    res_name=dataset_name+"_"+config["pretrained_model"]["pretrained_model_name"]
    lr0=dataset_cfg["lr"]
    lrf=dataset_cfg["lrf"]
    weigths_path = config["pretrained_model"]["pretrained_model_path"]
    data_yaml_path = dataset_cfg["data_yaml_path"]
    project_dir = config["path"]["train_result_path"]
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    epochs = dataset_cfg["epochs"]
    batch = dataset_cfg["batch_size"]
    imgsz = dataset_cfg["img_size"]
    patience=dataset_cfg["patience"]

    model = YOLO(str(weigths_path))

    model.train(
        data=str(data_yaml_path),
        project=str(project_dir),
        name=res_name,
        epochs=epochs,
        batch=batch,
        pretrained=True,
        imgsz=imgsz,
        lr0=lr0,
        lrf=lrf,
        patience=patience,
    )
    
    print(f"Training complete. Results stored under: {project_dir}")

if __name__ == "__main__":
    
    train_model()
