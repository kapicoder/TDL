from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import requests
from requests import Response


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """Assemble payload for /batch_split endpoint."""
    payload: Dict[str, Any] = {
        "images_dir": args.images_dir,
        "labels_dir": args.labels_dir,
        "weights": args.weights,
        "device": args.device,
        "no_visualize": args.no_visualize,
    }
    if args.output_dir:
        payload["output_dir"] = args.output_dir
        payload["patch_output_dir"] = args.output_dir
    if args.conf is not None:
        payload["conf"] = args.conf
    if args.iou is not None:
        payload["iou"] = args.iou
    if args.batch is not None:
        payload["batch"] = args.batch
    if args.imgsz is not None:
        payload["imgsz"] = args.imgsz
    return payload


def call_batch_split(
    base_url: str,
    payload: Dict[str, Any],
    verify_ssl: bool = True,
    timeout: int = 300,
) -> Response:
    """Send request to batch_split API and return response object."""
    url = f"{base_url.rstrip('/')}/batch_split"
    resp = requests.post(url, json=payload, verify=verify_ssl, timeout=timeout)
    resp.raise_for_status()
    return resp


def request_batch_split(
    *,
    base_url : str = "http://localhost:8000",
    images_dir: str = "dataset/AIR-SARShip-1.0/test/images",
    weights: str = "model/best.pt",
    labels_dir: str | None = None,
    output_dir: str | None = None,
    device: str = "cuda",
    no_visualize: bool = False,
    conf: float = 0.35,
    iou: float = 0.6,
    batch: int = 32,
    tile_size: int = 512,
    verify_ssl: bool = True,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    高层封装：通过函数传参构造 payload 并发起请求，返回解析后的 JSON。

    Args:
        base_url: API 服务基础地址（包含协议和端口），例如 `http://localhost:8000`。
        images_dir: 容器内影像目录或单个影像文件路径。
        weights: 容器内模型权重路径。
        labels_dir: 可选，容器内原始标签目录。
        output_dir: 可选，容器内输出目录，如果不存在会自动创建。默认存放在results/test_results下
        device: 推理设备，如 `cuda` 或 `cpu`。
        no_visualize: 是否禁用可视化,如果启用这个的话，只会输出预测的txt文件，其中只包含预测的坐标以及类别，没有经纬度。
        conf: 置信度阈值，默认 0.35。
        iou: IoU 阈值，默认 0.6。
        batch: 推理批大小，默认 32。
        tile_size: 推理输入尺寸，默认 256，如果输入图像大于这个，则会进行裁剪。
        verify_ssl: HTTPS 时是否校验证书（自签证书可设为 False）。
        timeout: 请求超时（秒）。

    Raises:
        requests.HTTPError: 当 HTTP 状态码非 2xx。
        requests.RequestException: 网络等其他请求错误。
        json.JSONDecodeError: 返回体不是 JSON。
    """
    payload: Dict[str, Any] = {
        "images_dir": images_dir,
        "weights": weights,
        "device": device,
        "no_visualize": no_visualize,
    }
    if labels_dir:
        payload["labels_dir"] = labels_dir
    if output_dir:
        payload["output_dir"] = output_dir
        payload["patch_output_dir"] = output_dir
    payload["conf"] = conf
    payload["iou"] = iou
    payload["batch"] = batch
    payload["tile_size"] = tile_size

    resp = call_batch_split(
        base_url=base_url,
        payload=payload,
        verify_ssl=verify_ssl,
        timeout=timeout,
    )
    return resp.json()
if __name__ == "__main__":
    request_batch_split(tile_size=640,output_dir="./TDL_results",no_visualize=False)
