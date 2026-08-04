#!/usr/bin/env python3
"""Reject private paths, generated outputs and large artifacts before release."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {
    "docs",
    "outputs",
    "output",
    "search_results",
    "best_engines",
    "codex_handoffs",
    "_001",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".cache",
    ".ckpt",
    ".engine",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".plan",
    ".pt",
    ".pth",
    ".safetensors",
    ".so",
    ".trt",
    ".weights",
}
FORBIDDEN_PATH_PREFIXES = tuple(
    "/" + name + "/" for name in ("home", "data", "root")
) + ("/" + "path" + "/" + "to",)


def release_files(root: Path) -> list[Path]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        root / value.decode("utf-8")
        for value in completed.stdout.split(b"\0")
        if value
    ]


def audit(root: Path, *, maximum_bytes: int) -> list[str]:
    failures: list[str] = []
    for path in release_files(root):
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        if FORBIDDEN_PARTS.intersection(relative.parts):
            failures.append(f"generated_output:{relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"binary_artifact:{relative}")
        if path.stat().st_size > maximum_bytes:
            failures.append(f"oversized_file:{relative}:{path.stat().st_size}")
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for prefix in FORBIDDEN_PATH_PREFIXES:
            if prefix in source:
                failures.append(f"absolute_private_path:{relative}:{prefix}")
    return sorted(set(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--maximum-mib", type=float, default=8.0)
    args = parser.parse_args(argv)
    failures = audit(
        args.root.resolve(),
        maximum_bytes=int(float(args.maximum_mib) * 1024 * 1024),
    )
    if failures:
        print("公开发布检查失败:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2
    print("公开发布检查通过：未发现输出、模型、引擎、大文件或私有绝对路径。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
