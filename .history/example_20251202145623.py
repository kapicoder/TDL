from __future__ import annotations

import json
import sys
import time

from TDL_docker_run import start_docker, stop_docker
from batch_split_request import request_batch_split


def main() -> None:
    container_name = "example"
    # 启动容器
    ok, msg = start_docker(
        host_port=8000,
        result_dir="./docker_result",
        dataset_dir="./dataset",
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
        resp = request_batch_split(

        )
        print("API 返回：")
        print(json.dumps(resp, ensure_ascii=False, indent=2))
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
