from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from flask import Flask, jsonify, request

from batch_split import _build_fast_split_args, batch_split

app = Flask(__name__)


def _to_path(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser().resolve()


def _build_batch_args(payload: Dict[str, Any]) -> argparse.Namespace:
    """将请求体转换为 batch_split 需要的 argparse.Namespace."""

    def _get(key: str, default: Any) -> Any:
        return payload.get(key, default)

    # 与 batch_split.parse_args 中的默认值保持一致；API 默认关闭可视化
    return argparse.Namespace(
        images_dir=_to_path(_get("images_dir", None)),
        labels_dir=_to_path(_get("labels_dir", None)),
        ext=_get("extensions", [".tif", ".tiff"]),
        tile_size=_get("tile_size", None),
        overlap=_get("overlap", None),
        weights=_to_path(_get("weights", None)),
        conf=float(_get("conf", 0.35)),
        iou=float(_get("iou", 0.6)),
        imgsz=int(_get("imgsz", 640)),
        batch=int(_get("batch", 32)),
        device=_get("device", "cuda"),
        half=bool(_get("half", False)),
        merge_iou=float(_get("merge_iou", 0.6)),
        output_dir=_to_path(_get("output_dir", None)),
        patch_output_dir=_to_path(_get("patch_output_dir", None)),
        patch_size=_get("patch_size", None),
        no_visualize=bool(_get("no_visualize", True)),
        visualize_original_label=bool(_get("visualize_original_label", False)),
        no_use_latlon=bool(_get("no_use_latlon", False)),
        no_export_detection_patches=bool(_get("no_export_detection_patches", False)),
        quiet=bool(_get("quiet", True)),
        block=bool(_get("block", False)),
    )


def _validate_args(args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    if args.images_dir is None:
        errors.append("images_dir 为必填字段")
    return errors


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok"}), 200


@app.route("/batch_split", methods=["POST"])
def run_batch_split() -> Any:
    if not request.is_json:
        return jsonify({"error": "请求体必须为 JSON"}), 400

    payload: Dict[str, Any] = request.get_json(force=True)
    batch_args = _build_batch_args(payload)
    validation_errors = _validate_args(batch_args)
    if validation_errors:
        return jsonify({"error": "; ".join(validation_errors)}), 400

    # 将 API 请求转换为 fast_split 需要的参数
    fast_args = _build_fast_split_args(batch_args)

    try:
        batch_split(
            images_dir=batch_args.images_dir,  # type: ignore[arg-type]
            labels_dir=batch_args.labels_dir,
            extensions=batch_args.ext,
            fast_split_args=fast_args,
            block_visual=batch_args.block,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    return jsonify({"status": "done"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
