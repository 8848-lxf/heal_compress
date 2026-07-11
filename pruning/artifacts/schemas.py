"""Physical pruning artifact schema constants."""

SNAPSHOT_SCHEMA_VERSION = "physical-structure-snapshot-v2"
HASH_SCHEMA_VERSION = "physical-structure-v2"
LEDGER_SCHEMA_VERSION = "physical-pruning-application-ledger-v2"
SAMPLING_REQUEST_SCHEMA_VERSION = "sampling-pruning-request-v1"
TERMINAL_APPLICATION_STATUSES = frozenset({"applied", "repaired", "merged", "skipped"})

STRUCTURE_HASH_FIELDS = (
    "canonical_module_name",
    "canonical_order",
    "module_type",
    "groups",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "output_padding",
)
SHAPE_HASH_FIELDS = (
    "canonical_module_name",
    "module_type",
    "in_channels",
    "out_channels",
    "in_features",
    "out_features",
    "num_features",
    "groups",
    "weight_shape",
    "bias_shape",
)

__all__ = [
    "HASH_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "SAMPLING_REQUEST_SCHEMA_VERSION",
    "SHAPE_HASH_FIELDS",
    "SNAPSHOT_SCHEMA_VERSION",
    "STRUCTURE_HASH_FIELDS",
    "TERMINAL_APPLICATION_STATUSES",
]
