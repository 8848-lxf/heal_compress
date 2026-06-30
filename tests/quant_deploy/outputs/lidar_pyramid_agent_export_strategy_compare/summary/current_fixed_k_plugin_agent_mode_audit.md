# Current Fixed-K Plugin Agent Mode Audit

field | value
--- | ---
current_agent_export_mode | padded_agent_static_fixed_k_scatter_plugin
current_max_cav | 2
uses_valid_agent_mask | True
uses_dynamic_agent_dim | False
uses_padded_agent_static | True
record_len_exists | False
valid_agent_mask_exists | True
pairwise_t_matrix_shape | `["batch", "max_cav", "max_cav", 4, 4]`
evidence | fixed-K plugin ONNX has valid_agent_mask input and no record_len input; plugin output uses static num_agents.
