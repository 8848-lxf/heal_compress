"""Grouped-convolution legality expressed in channels per group."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import DEFAULT_ALLOWED_CHANNELS_PER_GROUP
from ..exceptions import GroupedConvLegalityError
from ..types import GroupedConvShapeReport


def validate_grouped_conv_shape(
    *,
    in_channels: int,
    out_channels: int,
    groups: int,
    allowed_channels_per_group: Sequence[int] = DEFAULT_ALLOWED_CHANNELS_PER_GROUP,
    depthwise_special_case: bool = True,
    raise_on_error: bool = False,
) -> GroupedConvShapeReport:
    """Validate divisibility and the discrete per-group width set.

    The allowed set applies independently to logical input channels per group
    and logical output channels per group. A true depthwise convolution is a
    separate legal case and therefore is not rejected because its width per
    group equals one.
    """

    cin, cout, group_count = int(in_channels), int(out_channels), int(groups)
    allowed = tuple(sorted({int(value) for value in allowed_channels_per_group}))
    violations: list[str] = []
    if cin <= 0 or cout <= 0 or group_count <= 0:
        violations.append("channels_and_groups_must_be_positive")
    if group_count > 0 and cin % group_count:
        violations.append("in_channels_not_divisible_by_groups")
    if group_count > 0 and cout % group_count:
        violations.append("out_channels_not_divisible_by_groups")
    in_per = cin // group_count if group_count > 0 and cin % group_count == 0 else None
    out_per = cout // group_count if group_count > 0 and cout % group_count == 0 else None
    depthwise = bool(group_count > 1 and cin == cout == group_count)
    if not (depthwise and depthwise_special_case):
        if in_per is not None and in_per not in allowed:
            violations.append("input_channels_per_group_not_allowed")
        if out_per is not None and out_per not in allowed:
            violations.append("output_channels_per_group_not_allowed")
    common = in_per if in_per == out_per else None
    report = GroupedConvShapeReport(
        legal=not violations,
        in_channels=cin,
        out_channels=cout,
        groups=group_count,
        in_channels_per_group=in_per,
        out_channels_per_group=out_per,
        channels_per_group=common,
        allowed_channels_per_group=allowed,
        violations=violations,
    )
    if violations and raise_on_error:
        raise GroupedConvLegalityError(
            "invalid grouped convolution: " + ", ".join(violations)
        )
    return report


def validate_remove_groups(
    *,
    groups_before: int,
    kept_groups: Sequence[int],
    in_channels_per_group: int,
    out_channels_per_group: int,
) -> int:
    """Validate explicit whole-group removal and return the new group count."""

    keep = sorted({int(value) for value in kept_groups})
    if not keep or keep[0] < 0 or keep[-1] >= int(groups_before):
        raise GroupedConvLegalityError("remove_groups requires a non-empty valid group block set")
    if keep != list(range(keep[0], keep[-1] + 1)):
        raise GroupedConvLegalityError("remove_groups requires a contiguous group block")
    report = validate_grouped_conv_shape(
        in_channels=len(keep) * int(in_channels_per_group),
        out_channels=len(keep) * int(out_channels_per_group),
        groups=len(keep),
        raise_on_error=False,
    )
    if not report.legal:
        raise GroupedConvLegalityError("illegal shape after remove_groups: " + ", ".join(report.violations))
    return len(keep)


__all__ = ["validate_grouped_conv_shape", "validate_remove_groups"]
