"""RTX 4090-only contracts for high-order Transformer head alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def alignment_traits(d_h: int, *, heads: int, original_d_h: int) -> dict[str, Any]:
    width = int(d_h)
    original = int(original_d_h)
    head_count = int(heads)
    if width <= 0 or width > original or head_count <= 0:
        raise ValueError("invalid_power_alignment_width")
    projection = head_count * width
    result: dict[str, Any] = {
        "d_h": width,
        "heads": head_count,
        "original_d_h": original,
        "projection_width": projection,
        "exact_power_of_two": _is_power_of_two(width),
        "reduction_ratio": 1.0 - width / original,
    }
    for divisor in (4, 8, 16, 32, 64):
        result[f"divisible_by_{divisor}"] = width % divisor == 0
        result[f"projection_divisible_by_{divisor}"] = projection % divisor == 0
    return result


@dataclass(frozen=True)
class PowerAlignmentCandidate:
    d_h: int
    heads: int
    original_d_h: int
    exact_power_of_two_member: bool
    multiple_of_8_ladder_member: bool
    nearby_4_control_member: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            **alignment_traits(self.d_h, heads=self.heads, original_d_h=self.original_d_h),
            "exact_power_of_two_member": self.exact_power_of_two_member,
            "multiple_of_8_ladder_member": self.multiple_of_8_ladder_member,
            "nearby_4_control_member": self.nearby_4_control_member,
        }


def power_alignment_widths(original_d_h: int, *, heads: int) -> tuple[PowerAlignmentCandidate, ...]:
    original = int(original_d_h)
    if original not in {16, 32, 64}:
        raise ValueError(f"unsupported_power_alignment_original_width:{original}")
    powers = {value for value in (4, 8, 16, 32, 64) if value <= original}
    ladder = set(range(original, 7, -8))
    controls = {
        neighbor
        for target in ladder
        for neighbor in (target - 4, target + 4)
        if 0 < neighbor <= original
    }
    widths = sorted(powers | ladder | controls, reverse=True)
    return tuple(
        PowerAlignmentCandidate(
            d_h=width,
            heads=int(heads),
            original_d_h=original,
            exact_power_of_two_member=width in powers,
            multiple_of_8_ladder_member=width in ladder,
            nearby_4_control_member=width in controls,
        )
        for width in widths
    )


def candidate_width_values(rows: tuple[PowerAlignmentCandidate, ...]) -> tuple[int, ...]:
    return tuple(int(row.d_h) for row in rows)


def neighbor_controls(target_d_h: int, original_d_h: int) -> tuple[int, ...]:
    target = int(target_d_h)
    original = int(original_d_h)
    if target <= 0 or target > original:
        raise ValueError("invalid_neighbor_control_target")
    return tuple(value for value in (target - 4, target + 4) if 0 < value <= original)


@dataclass(frozen=True)
class JointAlignmentCandidate:
    candidate_id: str
    model: str
    target_d_h_by_family: dict[str, int]
    alignment_class: str
    diagnostic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_COBEVT_FAMILIES = ("cobevt_grid_h8_d32", "cobevt_window_h8_d32")
_V2XVIT_FAMILIES = (
    "v2xvit_agent_relation_h8_d32",
    "v2xvit_spatial_window_w16_h4_d64",
    "v2xvit_spatial_window_w4_h16_d16",
    "v2xvit_spatial_window_w8_h8_d32",
)


def joint_candidates(model: str) -> tuple[JointAlignmentCandidate, ...]:
    if model == "lidar_cobevt":
        widths = (
            (32, 32), (24, 24), (24, 16), (16, 24), (16, 16),
            (16, 8), (8, 16), (8, 8), (4, 4),
        )
        return tuple(
            JointAlignmentCandidate(
                candidate_id=f"C{index}",
                model=model,
                target_d_h_by_family=dict(zip(_COBEVT_FAMILIES, values)),
                alignment_class="diagnostic_4_aligned" if index == 8 else "high_alignment",
                diagnostic=index == 8,
            )
            for index, values in enumerate(widths)
        )
    if model == "lidar_v2xvit":
        widths = (
            (32, 64, 16, 32),
            (24, 56, 16, 24),
            (16, 48, 16, 16),
            (24, 48, 8, 24),
            (16, 32, 8, 16),
            (8, 24, 8, 8),
            (8, 16, 4, 8),
            (4, 8, 4, 4),
        )
        return tuple(
            JointAlignmentCandidate(
                candidate_id=f"V{index}",
                model=model,
                target_d_h_by_family=dict(zip(_V2XVIT_FAMILIES, values)),
                alignment_class="diagnostic_4_aligned" if index == 7 else "high_alignment",
                diagnostic=index == 7,
            )
            for index, values in enumerate(widths)
        )
    raise ValueError(f"unsupported_power_alignment_joint_model:{model}")


def speedup_metrics(
    *, baseline_p32_ms: float, baseline_profile_ms: float, candidate_profile_ms: float
) -> dict[str, float]:
    values = tuple(float(value) for value in (baseline_p32_ms, baseline_profile_ms, candidate_profile_ms))
    if any(value <= 0 for value in values):
        raise ValueError("latency_must_be_positive")
    p32, profile, candidate = values
    return {
        "structure_speedup": profile / candidate,
        "precision_speedup": p32 / profile,
        "total_speedup": p32 / candidate,
    }


def latency_benefit_gate(
    *, baseline_p50_ms: float, candidate_p50_ms: float,
    baseline_repeat_cv: float, baseline_replay_drift: float,
) -> dict[str, Any]:
    baseline = float(baseline_p50_ms)
    candidate = float(candidate_p50_ms)
    if baseline <= 0 or candidate <= 0:
        raise ValueError("latency_must_be_positive")
    required = max(0.01, 3.0 * float(baseline_repeat_cv), float(baseline_replay_drift))
    observed = 1.0 - candidate / baseline
    return {
        "required_reduction_ratio": required,
        "observed_reduction_ratio": observed,
        "beneficial": observed > required,
    }


def neighbor_advantage(
    *, candidate_ms: float, lower_control_ms: float | None, upper_control_ms: float | None
) -> dict[str, Any]:
    controls = [float(value) for value in (lower_control_ms, upper_control_ms) if value is not None]
    if not controls:
        return {"advantage": False, "reason": "neighbor_controls_missing"}
    candidate = float(candidate_ms)
    return {
        "advantage": all(candidate < value for value in controls),
        "control_count": len(controls),
        "minimum_control_speedup": min(value / candidate for value in controls),
    }


def search_candidate_gate(
    *, fixed500_acceptable: bool, same_profile_latency: bool,
    neighbor_advantage_passed: bool, build_repeat_stable: bool, joint_supported: bool,
) -> dict[str, Any]:
    checks = (
        (fixed500_acceptable, "fixed500_accuracy_not_acceptable"),
        (same_profile_latency, "same_profile_latency_not_supported"),
        (neighbor_advantage_passed, "neighbor_control_advantage_missing"),
        (build_repeat_stable, "independent_build_not_stable"),
        (joint_supported, "joint_candidate_not_supported"),
    )
    reasons = [reason for passed, reason in checks if not passed]
    return {"search_space_candidate": not reasons, "reasons": reasons}


__all__ = [
    "JointAlignmentCandidate",
    "PowerAlignmentCandidate",
    "alignment_traits",
    "candidate_width_values",
    "joint_candidates",
    "latency_benefit_gate",
    "neighbor_advantage",
    "neighbor_controls",
    "power_alignment_widths",
    "search_candidate_gate",
    "speedup_metrics",
]
