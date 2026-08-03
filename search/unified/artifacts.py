"""Publish the selected deployable engine into a stable isolated directory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping


_ENGINE_SUFFIXES = {".plan", ".engine", ".trt"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _engine_paths(value: Any) -> list[Path]:
    rows: list[Path] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in {"engine_path", "plan_path", "best_engine_path"} and child:
                rows.append(Path(str(child)).expanduser())
            else:
                rows.extend(_engine_paths(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            rows.extend(_engine_paths(child))
    return rows


@dataclass(frozen=True)
class EnginePublication:
    status: str
    source: str = ""
    destination: str = ""
    sha256: str = ""
    mode: str = ""


class BestEnginePublisher:
    def __init__(self, output_dir: str | Path, *, mode: str = "hardlink") -> None:
        if mode not in {"hardlink", "copy"}:
            raise ValueError(f"unsupported_best_engine_publish_mode:{mode}")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.mode = mode

    def publish(
        self,
        result: Mapping[str, Any],
        *,
        family_id: str,
        candidate_id: str = "best",
    ) -> EnginePublication:
        sources = [
            path.resolve()
            for path in _engine_paths(result)
            if path.is_file() and path.suffix.lower() in _ENGINE_SUFFIXES
        ]
        if not sources:
            return EnginePublication(status="no_deployable_engine_in_result")
        source = sources[0]
        destination_dir = self.output_dir / family_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{candidate_id}{source.suffix.lower()}"
        if destination.exists():
            if _sha256(destination) != _sha256(source):
                raise RuntimeError(f"best_engine_destination_conflict:{destination}")
        elif self.mode == "hardlink":
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        else:
            shutil.copy2(source, destination)
        publication = EnginePublication(
            status="published",
            source=str(source),
            destination=str(destination),
            sha256=_sha256(destination),
            mode=self.mode,
        )
        manifest = destination_dir / "best_engine_manifest.json"
        manifest.write_text(
            json.dumps(asdict(publication), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return publication


__all__ = ["BestEnginePublisher", "EnginePublication"]
