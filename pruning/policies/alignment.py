"""Dense-channel alignment policy."""

from __future__ import annotations

from ..types import AlignmentRepair


def legalize_dense_keep_count(
    channels_before: int,
    requested_keep: int,
    *,
    alignment: int = 4,
    fixed_output_contract: bool = False,
    minimum_retained_channels: int = 1,
) -> AlignmentRepair:
    """Round a mutable dense width down to a legal closure-wide width.

    Fixed output contracts are never rounded. For mutable widths, rounding
    down prevents a budget repair from silently retaining more channels than
    requested. The minimum is respected when a legal aligned value exists.
    """

    before = int(channels_before)
    requested = max(0, min(before, int(requested_keep)))
    align = int(alignment)
    if before <= 0 or align <= 0:
        raise ValueError("channels_before and alignment must be positive")
    if fixed_output_contract:
        return AlignmentRepair(
            channels_before=before,
            requested_keep=requested,
            final_keep=before,
            alignment=align,
            repaired=requested != before,
            protection_preserved=True,
            reason="fixed_output_contract",
        )
    lower = (requested // align) * align if align > 1 else requested
    minimum = max(1, int(minimum_retained_channels))
    if lower < minimum:
        aligned_minimum = ((minimum + align - 1) // align) * align if align > 1 else minimum
        lower = aligned_minimum if aligned_minimum <= before else before
    final = max(1, min(before, lower))
    return AlignmentRepair(
        channels_before=before,
        requested_keep=requested,
        final_keep=final,
        alignment=align,
        repaired=final != requested,
        reason="dense_conv_channel_alignment" if final != requested else "",
    )


__all__ = ["legalize_dense_keep_count"]
