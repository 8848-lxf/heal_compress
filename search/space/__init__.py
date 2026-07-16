"""Legal production search-space descriptions."""

from .legal_width_inventory import (
    LegalWidthDomain,
    LegalWidthInventory,
    build_legal_width_inventory,
    prepare_legal_width_search_space,
    write_legal_width_search_space_artifacts,
)

__all__ = [
    "LegalWidthDomain",
    "LegalWidthInventory",
    "build_legal_width_inventory",
    "prepare_legal_width_search_space",
    "write_legal_width_search_space_artifacts",
]
