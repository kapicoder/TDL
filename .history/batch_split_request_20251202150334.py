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
    if args.patch_output_dir:
        payload["patch_output_dir"] = args.patch_output_dir
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
    images_dir: str = "dataset/AIR-SARShip-1.0/test/images/SARShip-1.0-3.tiff",
    weights: str = "model/best.pt",
    labels_dir: str | None = None,
    output_dir: str | None = None,
    patch_output_dir: str | None = None,
    device: str = "cuda",
    no_visualize: bool = False,
    conf: float = 0.35,
    iou: float = 0.6,
    batch: int = 32,
    tile_size: int = 256,
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
        output_dir: 可选，容器内输出目录。
        patch_output_dir: 可选，容器内目标子图输出目录。
        device: 推理设备，如 `cuda` 或 `cpu`。
        no_visualize: 是否禁用可视化（默认 False，与 batch_split.py 对齐）。
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
    if patch_output_dir:
        payload["patch_output_dir"] = patch_output_dir
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTTPS client for batch_spilt_api.py /batch_split endpoint."
    )
    parser.add_argument(
        "--base-url",
        default="https://localhost:8000",
        help="API 服务基础地址（需包含协议），例如 https://your-host:8000",
    )
    parser.add_argument("--images-dir", required=True, help="待处理影像目录（容器内路径）")
    parser.add_argument("--labels-dir", help="原始标签目录（容器内路径，可选）")
    parser.add_argument("--weights", required=True, help="模型权重路径（容器内路径）")
    parser.add_argument("--output-dir", help="输出目录（容器内路径，可选）")
    parser.add_argument("--patch-output-dir", help="子图输出目录（容器内路径，可选）")
    parser.add_argument("--device", default="cuda", help="推理设备，如 cuda 或 cpu")
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="禁用可视化，默认开启可视化（与 batch_split.py 对齐）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="置信度阈值，默认 0.35（对齐 batch_split.py）",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU 阈值，默认 0.6（对齐 batch_split.py）",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="推理批大小，默认 32（对齐 batch_split.py）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推理输入尺寸，默认 640（对齐 batch_split.py）",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="跳过 HTTPS 证书校验（仅在内网/自签发证书时使用）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="请求超时（秒）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)

    try:
        resp = call_batch_split(
            base_url=args.base_url,
            payload=payload,
            verify_ssl=not args.insecure,
            timeout=args.timeout,
        )
    except requests.HTTPError as http_err:
        print(f"[HTTP error] {http_err} | body: {http_err.response.text}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as req_err:
        print(f"[Request error] {req_err}", file=sys.stderr)
        sys.exit(1)

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(resp.text)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    request_batch_split(tile_size=640)
