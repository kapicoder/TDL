from __future__ import annotations

import argparse
import numpy as np
import tempfile
from pathlib import Path
import yaml
from ultralytics import YOLO  # type: ignore
import json
from utils.config import CONFIG

def train_model(config: CONFIG
) -> None:
    """Train YOLO on MAR_reset using the provided pretrained weights."""
    
    dataset_name=config["train_dataset"]


    res_name=dataset_name+"_"+config["pretrained_model_name"]
    lr0=config["lr"]
    lrf=config["lrf"]
    weigths_path = config["pretrained_model_path"]
    data_yaml_path = config["train_data_yaml_path"]
    project_dir = config["train_result_path"]
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    epochs = config["epochs"]
    batch = config["batch_size"]
    imgsz = config["img_size"]
    patience=config["patience"]

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
    cfg=CONFIG()
    train_model(cfg)
