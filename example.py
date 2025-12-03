from __future__ import annotations

import json
import sys
import time
import subprocess

from TDL_docker_run import start_docker, stop_docker
from batch_split_request import request_batch_split


def main() -> None:
    container_name = "example"
    # 启动容器
    ok, msg = start_docker(
        host_port=8000,
        host_dir="../target_detection_location",
        name=container_name,
        image="ultralytics/ultralytics:latest"
    )
    if not ok:
        print(f"启动失败: {msg}", file=sys.stderr)
        return
    print(f"容器启动成功: {msg}")

    # 简单等待服务就绪
    time.sleep(5)

    # 调用 batch_split API
    try:
        print("开始请求调用并执行api")
        resp = request_batch_split(images_dir="dataset/AIR-SARShip-1.0/test/images")
        print("API 返回：")
        print(json.dumps(resp, ensure_ascii=False, indent=2))
        # 输出容器日志到终端
        try:
            logs = subprocess.run(
                ["docker", "logs", container_name],
                check=True,
                capture_output=True,
                text=True,
            )
            print("\n--- 容器日志 ---")
            print(logs.stdout)
        except subprocess.CalledProcessError as log_err:
            print(f"[获取日志失败] {log_err.stderr or log_err}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"请求失败: {exc}", file=sys.stderr)
    finally:
        ok_stop, msg_stop = stop_docker(container_name)
        if ok_stop:
            print(f"容器已停止: {msg_stop}")
        else:
            print(f"停止容器失败: {msg_stop}", file=sys.stderr)


if __name__ == "__main__":
    main()
