from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request

from test_batch import run_batch_test
from utils.config import CONFIG

app = Flask(__name__)
"""
对定位的api封装
"""

def _to_path(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser().resolve()


def _build_config(payload: Dict[str, Any]) -> tuple[CONFIG, Path | None, bool]:
    cfg = CONFIG()
    images_dir = _to_path(payload.get("images_dir"))
    weights = _to_path(payload.get("weights"))
    visualization_patch_output_dir = _to_path(payload.get("visualization_patch_output_dir"))
    conf = payload.get("conf")
    iou = payload.get("iou")
    device = payload.get("device")
    patch_size = payload.get("patch_size")
    test_visualize = bool(payload.get("test_visualize", False))
    no_label = bool(payload.get("no_label", False))
    show_pred_panel = payload.get("show_pred_panel", None)
    cfg.update_config(
        batch_test_img_path=images_dir,
        test_visualize=test_visualize,
        test_weights_path=weights,
        test_output_path=visualization_patch_output_dir,
        test_conf=conf,
        test_iou=iou,
        test_device=device,
        visualization_patch_output_dir=visualization_patch_output_dir,
        target_patch_size=patch_size,
        show_pred_panel=show_pred_panel,
    )
    if no_label:
        cfg.update_config(use_label=False)

    return cfg


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok"}), 200


@app.route("/batch_split", methods=["POST"])
def run_batch_split() -> Any:
    if not request.is_json:
        return jsonify({"error": "请求体必须为 JSON"}), 400

    payload: Dict[str, Any] = request.get_json(force=True)
    config = _build_config(payload)

    # try:
    results = run_batch_test(
        config=config,
    )
    if not results:
        return jsonify({"error": "未找到符合要求的图像"}), 400
    # except Exception as exc:  # noqa: BLE001
    #     return jsonify({"error": str(exc)}), 500

    def _to_str(val: Any) -> Any:
        if isinstance(val, Path):
            return str(val)
        return val

    response_results: List[Dict[str, Any]] = []
    for item in results:
        response_results.append({k: _to_str(v) for k, v in item.items()})

    return jsonify({"status": "done", "processed": len(results), "results": response_results}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
