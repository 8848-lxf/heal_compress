# Dynamic Agent Dim Scatter Plugin Feasibility

field | value
--- | ---
dynamic_agent_dim_single_engine_supported | False
dynamic_agent_dim_per_N_engine_required | True
plugin_output_agent_dim_source | serialized plugin attribute num_agents
getOutputDimensions_uses_fixed_num_agents | True
enqueue_uses_fixed_num_agents | True
required_plugin_changes | `["To support true dynamic N in one engine, add an input or shape-tensor source for N and return that dimension in getOutputDimensions.", "Update enqueue to derive numAgents from runtime shape/input instead of mParams.numAgents.", "Revalidate TensorRT dynamic output dimensions and bucket profiles for pairwise_t_matrix [1,N,N,4,4]."]`
selected_test_plan | Use per-N fallback engines dynamic_agent_dim_N1 and dynamic_agent_dim_N2; do not silently use padded max_cav=2 for this comparison.
