"""Build a complete terminal ledger for every sampling request."""

from __future__ import annotations

from ..types import (
    ApplicationLedgerEntry,
    PhysicalPruningApplicationLedger,
    PhysicalPruningPlan,
)


def build_application_ledger(plan: PhysicalPruningPlan) -> PhysicalPruningApplicationLedger:
    """Record applied/repaired/merged/skipped status for every request id."""

    request_lookup = {
        row.request_id: row
        for row in (plan.source_request.entries if plan.source_request is not None else [])
    }
    entries: list[ApplicationLedgerEntry] = []
    covered: set[str] = set()
    for plan_entry in plan.entries:
        request_ids = list(plan_entry.source_request_ids)
        primary = request_ids[0] if request_ids else ""
        for request_id in request_ids:
            source = request_lookup.get(request_id)
            is_primary = request_id == primary
            if not is_primary:
                status = "merged"
            elif plan_entry.repaired:
                status = "repaired"
            else:
                status = "applied"
            entries.append(
                ApplicationLedgerEntry(
                    request_id=request_id,
                    status=status,
                    module_path=plan_entry.module_path,
                    axis=plan_entry.axis,
                    requested_prune_indices=list(source.prune_indices if source else plan_entry.prune_indices),
                    applied_prune_indices=list(plan_entry.prune_indices),
                    applied_keep_indices=list(plan_entry.keep_indices),
                    reason=plan_entry.repair_reason,
                    merged_into_request_id=primary if status == "merged" else "",
                    alignment=dict(plan_entry.metadata.get("alignment_repair", {})),
                    protection={
                        "fixed_output_contract": bool(plan_entry.metadata.get("fixed_output_contract")),
                        "dependency_driven": bool(plan_entry.metadata.get("dependency_driven")),
                        "reason": str(plan_entry.metadata.get("protection_reason", "")),
                    },
                    closure={"scope_ids": list(plan_entry.metadata.get("scope_ids", []))},
                )
            )
            covered.add(request_id)
    for request_id, source in request_lookup.items():
        if request_id in covered:
            continue
        entries.append(
            ApplicationLedgerEntry(
                request_id=request_id,
                status="skipped",
                module_path=source.module_path,
                axis=source.axis,
                requested_prune_indices=list(source.prune_indices),
                reason="empty_or_legalized_away_request",
            )
        )
    entries.sort(key=lambda row: row.request_id)
    return PhysicalPruningApplicationLedger(entries=entries)


__all__ = ["build_application_ledger"]
