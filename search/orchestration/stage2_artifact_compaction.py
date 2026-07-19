"""Lossless hardlink compaction for completed Stage-2 generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..artifacts.candidate_audit_store import (
    CandidateAuditStore,
    finalize_completed_candidate,
)


_IDENTICAL_ARTIFACT_GROUPS = (
    (
        "physical_pruning_plan.json",
        "physical_plan.json",
        "legalized_plan.json",
    ),
    ("pruning_request.json", "sampling_pruning_request.json"),
    ("materialization_report.json", "materialization_ledger.json"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _content_addressed_candidates(
    retention_markers: list[Path],
) -> list[tuple[Path, Path]]:
    candidates = []
    for marker in retention_markers:
        stage2_dir = marker.parent / "stage2"
        if not stage2_dir.is_dir():
            continue
        for candidate_dir in sorted(stage2_dir.iterdir()):
            if not candidate_dir.is_dir() or candidate_dir.is_symlink():
                continue
            has_plan = any(
                (candidate_dir / name).is_file()
                for name in (
                    "physical_pruning_plan.json",
                    "physical_plan.json",
                    "legalized_plan.json",
                )
            )
            has_request = any(
                (candidate_dir / name).is_file()
                for name in (
                    "pruning_request.json",
                    "sampling_pruning_request.json",
                )
            )
            if has_plan and has_request:
                candidates.append((candidate_dir, marker))
    return candidates


def _compact_content_addressed(
    destination: Path,
    retention_markers: list[Path],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    candidates = _content_addressed_candidates(retention_markers)
    reports = []
    if not dry_run:
        store = CandidateAuditStore(destination / "stage2_audit_store")
        for candidate_dir, marker in candidates:
            reports.append(
                finalize_completed_candidate(
                    candidate_dir,
                    store=store,
                    completion_marker=marker,
                )
            )
    summary = {
        "mode": "content_addressed",
        "root": str(destination),
        "dry_run": bool(dry_run),
        "completed_generation_count": len(retention_markers),
        "eligible_candidate_count": len(candidates),
        "compacted_candidate_count": sum(
            row.get("status") in {"compacted", "already_compacted"}
            for row in reports
        ),
        "removed_file_count": sum(
            int(row.get("removed_file_count", 0)) for row in reports
        ),
        "bytes_reclaimed": sum(
            int(row.get("removed_bytes", 0)) for row in reports
        ),
        "candidate_reports": reports,
    }
    _write_json(destination / "stage2_content_addressed_compaction.json", summary)
    return summary


def compact_run(
    root: str | Path,
    *,
    content_addressed: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compact known aliases in completed generations only."""

    destination = Path(root).expanduser().resolve()
    if not destination.is_dir():
        raise RuntimeError(f"stage2_compaction_root_missing:{destination}")
    retention_markers = sorted(
        destination.rglob("stage2_artifact_retention.json")
    )
    if content_addressed:
        return _compact_content_addressed(
            destination, retention_markers, dry_run=dry_run
        )
    if dry_run:
        raise ValueError("dry_run_requires_content_addressed_mode")
    linked_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    already_linked_count = 0

    for marker in retention_markers:
        generation_dir = marker.parent
        stage2_dir = generation_dir / "stage2"
        if not stage2_dir.is_dir():
            continue
        for candidate_dir in sorted(stage2_dir.iterdir()):
            if not candidate_dir.is_dir() or candidate_dir.is_symlink():
                continue
            for group in _IDENTICAL_ARTIFACT_GROUPS:
                canonical = candidate_dir / group[0]
                if not canonical.is_file() or canonical.is_symlink():
                    continue
                canonical_stat = canonical.stat()
                canonical_hash: str | None = None
                for alias_name in group[1:]:
                    alias = candidate_dir / alias_name
                    if not alias.is_file() or alias.is_symlink():
                        continue
                    alias_stat = alias.stat()
                    if (
                        canonical_stat.st_dev == alias_stat.st_dev
                        and canonical_stat.st_ino == alias_stat.st_ino
                    ):
                        already_linked_count += 1
                        continue
                    if canonical_stat.st_size != alias_stat.st_size:
                        mismatch_rows.append(
                            {
                                "canonical": str(canonical.relative_to(destination)),
                                "alias": str(alias.relative_to(destination)),
                                "reason": "size_mismatch",
                            }
                        )
                        continue
                    if canonical_hash is None:
                        canonical_hash = _sha256(canonical)
                    alias_hash = _sha256(alias)
                    if alias_hash != canonical_hash:
                        mismatch_rows.append(
                            {
                                "canonical": str(canonical.relative_to(destination)),
                                "alias": str(alias.relative_to(destination)),
                                "reason": "sha256_mismatch",
                            }
                        )
                        continue

                    temp = alias.with_name(f".{alias.name}.hardlink.tmp")
                    temp.unlink(missing_ok=True)
                    os.link(canonical, temp)
                    os.replace(temp, alias)
                    linked_rows.append(
                        {
                            "canonical": str(canonical.relative_to(destination)),
                            "alias": str(alias.relative_to(destination)),
                            "sha256": canonical_hash,
                            "size_bytes": int(alias_stat.st_size),
                            "allocated_bytes_reclaimed": int(
                                alias_stat.st_blocks * 512
                            ),
                        }
                    )

    summary = {
        "root": str(destination),
        "completed_generation_count": len(retention_markers),
        "linked_alias_count": len(linked_rows),
        "already_linked_count": already_linked_count,
        "content_mismatch_count": len(mismatch_rows),
        "bytes_reclaimed": sum(
            int(row["allocated_bytes_reclaimed"]) for row in linked_rows
        ),
        "linked_aliases": linked_rows,
        "content_mismatches": mismatch_rows,
    }
    _write_json(destination / "stage2_hardlink_compaction.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Losslessly hardlink duplicate completed Stage-2 artifacts."
    )
    parser.add_argument("root")
    parser.add_argument("--content-addressed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            compact_run(
                args.root,
                content_addressed=bool(args.content_addressed),
                dry_run=bool(args.dry_run),
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
