"""Small progress facade with Chinese phase messages and optional tqdm bars."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Iterable, Iterator, TypeVar


T = TypeVar("T")


@dataclass
class SearchProgress:
    enabled: bool = True
    stream: object = sys.stderr

    def phase(self, index: int, total: int, message: str) -> None:
        if self.enabled:
            print(f"[搜索阶段 {index}/{total}] {message}", file=self.stream, flush=True)

    def iterate(self, values: Iterable[T], *, description: str, total: int | None = None) -> Iterator[T]:
        if not self.enabled:
            yield from values
            return
        try:
            from tqdm.auto import tqdm
        except ImportError:
            for index, value in enumerate(values, start=1):
                suffix = f"/{total}" if total is not None else ""
                print(f"{description}: {index}{suffix}", file=self.stream, flush=True)
                yield value
            return
        yield from tqdm(values, desc=description, total=total, unit="步", file=self.stream)


__all__ = ["SearchProgress"]
