from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import requests
from requests import Response

from utils.config import CONFIG

"""
对 batch_split 接口的封装，提供命令行和函数调用两种方式。
"""



def build_payload(args: argparse.Namespace, cfg: CONFIG | None = None) -> Dict[str, Any]:
    """Assemble payload for /batch_split endpoint, with config defaults."""
    cfg = cfg or CONFIG()

    def choose(arg_val, cfg_key: str, fallback=None):
        if arg_val is not None:
            return arg_val
        cfg_val = cfg[cfg_key]
        return cfg_val if cfg_val is not None else fallback

    payload: Dict[str, Any] = {}
    images_dir = getattr(args, "images_dir", None)
    if images_dir:
        payload["images_dir"] = images_dir
    weights = choose(getattr(args, "weights", None), "test_weights_path")
    if weights:
        payload["weights"] = weights
    output_dir = choose(getattr(args, "output_dir", None), "test_output_path")
    if output_dir:
        payload["output_dir"] = output_dir
        payload["patch_output_dir"] = choose(getattr(args, "patch_output_dir", None), "patch_output_dir", output_dir)
    conf = choose(getattr(args, "conf", None), "test_conf", 0.25)
    iou = choose(getattr(args, "iou", None), "test_iou", 0.6)
    device = choose(getattr(args, "device", None), "test_device", cfg["device"] or "cuda")
    patch_size = choose(getattr(args, "patch_size", None), "target_patch_size")

    payload["conf"] = conf
    payload["iou"] = iou
    payload["device"] = device
    if patch_size is not None:
        payload["patch_size"] = patch_size
    payload["no_visualize"] = getattr(args, "no_visualize", False)
    payload["no_label"] = getattr(args, "no_label", False)
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
    images_dir: str | None = None,
    weights: str | None = None,
    output_dir: str | None = None,
    patch_output_dir: str | None = None,
    device: str | None = None,
    test_visualize: bool = False,
    no_label: bool = False,
    conf: float | None = None,
    iou: float | None = None,
    patch_size: int | None = None,
    verify_ssl: bool = True,
    timeout: int = 300,
    config: CONFIG | None = None,
) -> Dict[str, Any]:
    """
    高层封装：通过函数传参构造 payload 并发起请求，返回解析后的 JSON，默认使用 config.json 的配置。
    """
    cfg = CONFIG()
    payload: Dict[str, Any] = {}
    payload["images_dir"] = images_dir or cfg["batch_test_img_path"]
    payload["weights"] = weights or cfg["test_weights_path"]
    payload["visualization_patch_output_dir"] = output_dir or cfg["visualization_patch_output_dir"]
    payload["device"] = device or cfg["test_device"] or cfg["device"] or "cuda"
    payload["conf"] = conf if conf is not None else (cfg["test_conf"] or 0.25)
    payload["iou"] = iou if iou is not None else (cfg["test_iou"] or 0.6)
    if patch_size is not None or cfg["visualization_target_patch_size"] is not None:
        payload["visualization_target_patch_size"] = patch_size or cfg["visualization_target_patch_size"]
    payload["test_visualize"] = test_visualize
    payload["no_label"] = no_label

    resp = call_batch_split(
        base_url=base_url,
        payload=payload,
        verify_ssl=verify_ssl,
        timeout=timeout,
    )
    return resp.json()
if __name__ == "__main__":
    request_batch_split(output_dir="./TDL_results", test_visualize=False)
