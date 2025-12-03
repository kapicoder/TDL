from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple


def start_docker(
    host_port: int = 8000,
    result_dir: str = "./TDL_result",
    dataset_dir: str = "./dataset",
    *,
    image: str = "ultralytics/ultralytics:latest",
    name: str | None = None,
) -> Tuple[bool, str]:
    """
    启动封装好的 batch_split_api 容器。

    Args:
        host_port: 映射到容器 8000 的本地主机端口。
        result_dir: 宿主机 result 目录，挂载到容器 /workspace/TDL/result。
        dataset_dir: 宿主机 dataset 目录，挂载到容器 /workspace/TDL/dataset。
        image: 可选，自定义镜像名称。
        name: 可选，容器名称（便于停止）。

    Returns:
        (success, message)：success 表示启动是否成功，message 为容器 ID 或错误信息。
    """
    host_result = Path(result_dir).expanduser().resolve()
    host_dataset = Path(dataset_dir).expanduser().resolve()

    if not host_result.exists():
        return False, f"result_dir 不存在: {host_result}"
    if not host_dataset.exists():
        return False, f"dataset_dir 不存在: {host_dataset}"

    cmd = [
        "docker",
        "run",
        "--rm",
        "-d",
        "--gpus",
        "all",
        "-p",
        f"{host_port}:8000",
        "-w",
        "/workspace/TDL",
        "-v",
        f"{host_result}:/workspace/TDL/result",
        "-v",
        f"{host_dataset}:/workspace/TDL/dataset",
    ]
    if name:
        cmd.extend(["--name", name])
    cmd.extend(
        [
            image,
            "python",
            "batch_split_api.py",
        ]
    )

    try:
        proc = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or str(exc)
    return True, proc.stdout.strip()


def stop_docker(container: str) -> Tuple[bool, str]:
    """
    停止指定的容器。

    Args:
        container: 容器名称或 ID。

    Returns:
        (success, message)：success 表示停止是否成功，message 为输出或错误信息。
    """
    cmd = ["docker", "stop", container]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or str(exc)
    return True, proc.stdout.strip()


if __name__ == "__main__":
    ok, msg = start_docker(
        host_port=8000,
        result_dir="./TDL_result",
        dataset_dir="./dataset",
        name="tdl_api",
    )
    print(msg if ok else f"启动失败: {msg}")
