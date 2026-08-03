"""Legal pruning action search-space helpers."""

from .action_catalog import PruningActionCatalog, PruningSearchAction, build_pruning_action_catalog
from .grouped_bundle_adapter import request_from_pruning_actions
from .domain_importance import score_atomic_units_for_fixed_ranking
from .local_domains import (
    LocalPruningDomain,
    build_local_pruning_domains,
    expand_domain_width_genes,
    legalize_domain_width_genes,
)
from .transformer_domains import (
    AttentionInstanceSpec,
    FFNInstanceSpec,
    SharedTransformerParameterError,
    build_attention_dh_domain,
    build_ffn_hidden_domain,
    build_transformer_pruning_domains,
    discover_attention_instances,
    discover_ffn_instances,
    fixed_transformer_rankings_from_unit_scores,
    legal_attention_widths,
    legal_ffn_widths,
)
from .transformer_physical_pruner import (
    TransformerPhysicalPruneReport,
    materialize_transformer_widths,
    state_dict_shape_hash,
)
from .unified_physical_pruner import (
    CNN_DOMAIN_TYPES,
    TRANSFORMER_DOMAIN_TYPES,
    UnifiedPhysicalPruneReport,
    UnifiedPhysicalPruneResult,
    materialize_unified_widths,
)

__all__ = [
    "PruningActionCatalog",
    "PruningSearchAction",
    "build_pruning_action_catalog",
    "request_from_pruning_actions",
    "LocalPruningDomain",
    "build_local_pruning_domains",
    "expand_domain_width_genes",
    "legalize_domain_width_genes",
    "score_atomic_units_for_fixed_ranking",
    "AttentionInstanceSpec",
    "FFNInstanceSpec",
    "SharedTransformerParameterError",
    "build_attention_dh_domain",
    "build_ffn_hidden_domain",
    "build_transformer_pruning_domains",
    "discover_attention_instances",
    "discover_ffn_instances",
    "fixed_transformer_rankings_from_unit_scores",
    "legal_attention_widths",
    "legal_ffn_widths",
    "TransformerPhysicalPruneReport",
    "materialize_transformer_widths",
    "state_dict_shape_hash",
    "CNN_DOMAIN_TYPES",
    "TRANSFORMER_DOMAIN_TYPES",
    "UnifiedPhysicalPruneReport",
    "UnifiedPhysicalPruneResult",
    "materialize_unified_widths",
]
