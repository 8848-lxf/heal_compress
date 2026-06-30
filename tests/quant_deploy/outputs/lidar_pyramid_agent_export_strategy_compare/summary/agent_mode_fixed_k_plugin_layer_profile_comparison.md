# Agent Mode Fixed-K Plugin Layer Profile Comparison

profile | rank | latency_ms | category | layer_type | layer
--- | --- | --- | --- | --- | ---
padded_agent_static_fixed_k_plugin_fp32 | 1 | 0.7383 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.0/Conv + /shrink_conv/layers.0/double_conv/double_conv.1/Relu
padded_agent_static_fixed_k_plugin_fp32 | 2 | 0.4946 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.2/Conv + /shrink_conv/layers.0/double_conv/double_conv.3/Relu
padded_agent_static_fixed_k_plugin_fp32 | 3 | 0.4142 | BEV backbone Conv |  | /layer0/layer0.0/conv2_1/Conv + /layer0/layer0.0/relu_3/Relu
padded_agent_static_fixed_k_plugin_fp32 | 4 | 0.2734 | Other |  | __myl_TranReshMulAddReshMoveTranReluMaxr_myl3_19
padded_agent_static_fixed_k_plugin_fp32 | 5 | 0.2273 | PillarVFE / PFN |  | /pillar_vfe/pfn_layers_0/linear/MatMul_myl3_18
padded_agent_static_fixed_k_plugin_fp32 | 6 | 0.1239 | Other |  | PWN(/Mul)
padded_agent_static_fixed_k_plugin_fp32 | 7 | 0.1220 | BEV backbone Conv |  | /layer0/layer0.0/conv1/Conv + /layer0/layer0.0/relu/Relu
padded_agent_static_fixed_k_plugin_fp32 | 8 | 0.1219 | BEV backbone Conv |  | /layer0/layer0.1/conv2_1/Conv + /layer0/layer0.1/relu_3/Relu
padded_agent_static_fixed_k_plugin_fp32 | 9 | 0.1219 | BEV backbone Conv |  | /layer0/layer0.2/conv2_1/Conv + /layer0/layer0.2/relu_3/Relu
padded_agent_static_fixed_k_plugin_fp32 | 10 | 0.0993 | BEV backbone Conv |  | /layer1/layer1.0/conv2/Conv + /layer1/layer1.0/relu_1/Relu
padded_agent_static_fixed_k_plugin_fp32 | 11 | 0.0911 | BEV backbone Conv |  | /deblocks.2/deblocks.2.0/ConvTranspose + /deblocks.2/deblocks.2.1/BatchNormalization + /deblocks.2/deblocks.2.2/Relu
padded_agent_static_fixed_k_plugin_fp32 | 12 | 0.0881 | PointPillarScatterTRT |  | /PointPillarScatterTRT
padded_agent_static_fixed_k_plugin_fp32 | 13 | 0.0778 | BEV backbone Conv |  | /layer1/layer1.0/conv1/Conv + /layer1/layer1.0/relu/Relu
padded_agent_static_fixed_k_plugin_fp32 | 14 | 0.0768 | BEV backbone Conv |  | /layer0/layer0.1/conv2/Conv + /layer0/layer0.1/Add + /layer0/layer0.1/relu_1/Relu
padded_agent_static_fixed_k_plugin_fp32 | 15 | 0.0768 | BEV backbone Conv |  | /layer0/layer0.2/conv2/Conv + /layer0/layer0.2/Add + /layer0/layer0.2/relu_1/Relu
padded_agent_static_fixed_k_plugin_fp32 | 16 | 0.0758 | BEV backbone Conv |  | /layer0/layer0.2/conv1/Conv + /layer0/layer0.2/relu/Relu
padded_agent_static_fixed_k_plugin_fp32 | 17 | 0.0748 | BEV backbone Conv |  | /layer0/layer0.0/conv2/Conv
padded_agent_static_fixed_k_plugin_fp32 | 18 | 0.0748 | BEV backbone Conv |  | /layer0/layer0.1/conv1/Conv + /layer0/layer0.1/relu/Relu
padded_agent_static_fixed_k_plugin_fp32 | 19 | 0.0666 | BEV backbone Conv |  | /layer0/layer0.0/conv3/Conv + /layer0/layer0.0_1/Add + /layer0/layer0.0/relu_4/Relu
padded_agent_static_fixed_k_plugin_fp32 | 20 | 0.0636 | BEV backbone Conv |  | /layer1/layer1.3/conv2/Conv + /layer1/layer1.3/relu_1/Relu
padded_agent_static_fixed_k_plugin_fp16 | 1 | 0.2273 | Other |  | __myl_TranReshMulAddReshMoveTranReluMaxr_myl5_20
padded_agent_static_fixed_k_plugin_fp16 | 2 | 0.1874 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.0/Conv + /shrink_conv/layers.0/double_conv/double_conv.1/Relu
padded_agent_static_fixed_k_plugin_fp16 | 3 | 0.1270 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.2/Conv + /shrink_conv/layers.0/double_conv/double_conv.3/Relu
padded_agent_static_fixed_k_plugin_fp16 | 4 | 0.1065 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_CastReshMulMulMulNegExpAddDivAddMulNegExpAddDivAddMulNegExpAddDivAddMulGridCastGridCastGridEtc_myl73_8
padded_agent_static_fixed_k_plugin_fp16 | 5 | 0.0952 | PillarVFE / PFN |  | /pillar_vfe/pfn_layers_0/linear/MatMul_myl5_19
padded_agent_static_fixed_k_plugin_fp16 | 6 | 0.0727 | PointPillarScatterTRT |  | /PointPillarScatterTRT
padded_agent_static_fixed_k_plugin_fp16 | 7 | 0.0481 | BEV backbone Conv |  | /layer0/layer0.0/conv2_1/Conv + /layer0/layer0.0/relu_3/Relu
padded_agent_static_fixed_k_plugin_fp16 | 8 | 0.0481 | BEV backbone Conv |  | /layer0/layer0.2/conv2_1/Conv + /layer0/layer0.2/relu_3/Relu
padded_agent_static_fixed_k_plugin_fp16 | 9 | 0.0472 | BEV backbone Conv |  | /layer0/layer0.1/conv2_1/Conv + /layer0/layer0.1/relu_3/Relu
padded_agent_static_fixed_k_plugin_fp16 | 10 | 0.0348 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | Reformatting CopyNode for Output Tensor 0 to PWN(/Mul)
padded_agent_static_fixed_k_plugin_fp16 | 11 | 0.0297 | BEV backbone Conv |  | /layer1/layer1.0/conv2/Conv + /layer1/layer1.0/relu_1/Relu
padded_agent_static_fixed_k_plugin_fp16 | 12 | 0.0287 | BEV backbone Conv |  | /layer0/layer0.0/conv2/Conv
padded_agent_static_fixed_k_plugin_fp16 | 13 | 0.0287 | BEV backbone Conv |  | /layer0/layer0.1/conv2/Conv + /layer0/layer0.1/Add + /layer0/layer0.1/relu_1/Relu
padded_agent_static_fixed_k_plugin_fp16 | 14 | 0.0277 | BEV backbone Conv |  | /layer1/layer1.3/conv2/Conv + /layer1/layer1.3/relu_1/Relu
padded_agent_static_fixed_k_plugin_fp16 | 15 | 0.0276 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_GtrReplSeleReshCastReshReshGtrReshCastDivMulSubConcMul_myl5_17
padded_agent_static_fixed_k_plugin_fp16 | 16 | 0.0276 | BEV backbone Conv |  | /layer0/layer0.0/conv1/Conv + /layer0/layer0.0/relu/Relu
padded_agent_static_fixed_k_plugin_fp16 | 17 | 0.0275 | BEV backbone Conv |  | /layer1/layer1.1/conv2/Conv + /layer1/layer1.1/relu_1/Relu
padded_agent_static_fixed_k_plugin_fp16 | 18 | 0.0266 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_ReplReshReplReshReplReshIotaCastReshReplReshIotaCastReshReplReshConcCastConcCastConcCast_myl5_6
padded_agent_static_fixed_k_plugin_fp16 | 19 | 0.0266 | Other |  | __myl_SlicSum_myl5_14
padded_agent_static_fixed_k_plugin_fp16 | 20 | 0.0266 | BEV backbone Conv |  | /layer0/layer0.1/conv1/Conv + /layer0/layer0.1/relu/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 1 | 0.8714 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.2/Conv + /shrink_conv/layers.0/double_conv/double_conv.3/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 2 | 0.7413 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.0/Conv + /shrink_conv/layers.0/double_conv/double_conv.1/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 3 | 0.2888 | BEV backbone Conv |  | /layer1/layer1.2/conv2/Conv + /layer1/layer1.2/relu_1/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 4 | 0.2744 | Other |  | __myl_TranReshMulAddReshMoveTranReluMaxr_myl2_19
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 5 | 0.2284 | PillarVFE / PFN |  | /pillar_vfe/pfn_layers_0/linear/MatMul_myl2_18
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 6 | 0.0891 | BEV backbone Conv |  | /deblocks.2/deblocks.2.0/ConvTranspose + /deblocks.2/deblocks.2.1/BatchNormalization + /deblocks.2/deblocks.2.2/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 7 | 0.0656 | Other |  | __myl_MulSum_myl92_24
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 8 | 0.0645 | BEV backbone Conv |  | /layer0/layer0.0/conv2_1/Conv + /layer0/layer0.0/relu_3/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 9 | 0.0635 | BEV backbone Conv |  | /layer0/layer0.2/conv2_1/Conv + /layer0/layer0.2/relu_3/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 10 | 0.0625 | BEV backbone Conv |  | /layer0/layer0.1/conv2_1/Conv + /layer0/layer0.1/relu_3/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 11 | 0.0584 | BEV backbone Conv |  | /layer0/layer0.0/conv1/Conv + /layer0/layer0.0/relu/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 12 | 0.0573 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_GtrReplSeleReshCastReshReshGtrReshCastDivMulSubConcMul_myl2_17
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 13 | 0.0573 | Other |  | __myl_MoveGrid_myl92_17
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 14 | 0.0458 | BEV backbone Conv |  | /deblocks.1/deblocks.1.0/ConvTranspose + /deblocks.1/deblocks.1.1/BatchNormalization + /deblocks.1/deblocks.1.2/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 15 | 0.0430 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_ReplReshReplReshReplReshIotaCastReshReplReshIotaCastReshReplReshConcCastConcCastConcCast_myl2_6
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 16 | 0.0420 | BEV backbone Conv |  | /layer0/layer0.2/conv1/Conv + /layer0/layer0.2/relu/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 17 | 0.0420 | BEV backbone Conv |  | /layer1/layer1.0/conv1/Conv + /layer1/layer1.0/relu/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 18 | 0.0410 | BEV backbone Conv |  | /layer0/layer0.0/conv2/Conv
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 19 | 0.0410 | BEV backbone Conv |  | /layer0/layer0.1/conv2/Conv + /layer0/layer0.1/Add + /layer0/layer0.1/relu_1/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp32 | 20 | 0.0410 | BEV backbone Conv |  | /layer0/layer0.2/conv2/Conv + /layer0/layer0.2/Add + /layer0/layer0.2/relu_1/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 1 | 0.5116 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.0/Conv + /shrink_conv/layers.0/double_conv/double_conv.1/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 2 | 0.2222 | Other |  | __myl_TranReshMulAddReshMoveTranReluMaxr_myl3_20
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 3 | 0.1319 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.2/Conv + /shrink_conv/layers.0/double_conv/double_conv.3/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 4 | 0.1096 | PillarVFE / PFN |  | /pillar_vfe/pfn_layers_0/linear/MatMul_myl3_19
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 5 | 0.0707 | Other |  | __myl_MulSum_myl68_26
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 6 | 0.0399 | Other |  | __myl_MulSum_myl68_17
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 7 | 0.0389 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_GridCast_myl68_19
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 8 | 0.0389 | PointPillarScatterTRT |  | /PointPillarScatterTRT
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 9 | 0.0287 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_GtrReplSeleReshCastReshReshGtrReshCastDivMulSubConcMul_myl3_17
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 10 | 0.0279 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_ReplReshReplReshReplReshIotaCastReshReplReshIotaCastReshReplReshConcCastConcCastConcCast_myl3_6
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 11 | 0.0276 | Other |  | __myl_SlicSum_myl3_14
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 12 | 0.0266 | BEV backbone Conv |  | /layer0/layer0.0/conv2_1/Conv + /layer0/layer0.0/relu_3/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 13 | 0.0256 | BEV backbone Conv |  | /layer0/layer0.2/conv2_1/Conv + /layer0/layer0.2/relu_3/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 14 | 0.0249 | BEV backbone Conv |  | /layer0/layer0.1/conv2_1/Conv + /layer0/layer0.1/relu_3/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 15 | 0.0215 | Other |  | __myl_Repl_myl3_4
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 16 | 0.0215 | Other |  | __myl_MulSum_myl68_8
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 17 | 0.0184 | BEV backbone Conv |  | /layer0/layer0.0/conv1/Conv + /layer0/layer0.0/relu/Relu
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 18 | 0.0184 | PillarVFE / PFN |  | Reformatting CopyNode for Input Tensor 0 to {ForeignNode[ONNXTRT_ShapeShuffle_342_output[Constant].../pillar_vfe/Squeeze]}
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 19 | 0.0174 | Other |  | __myl_MoveConc_myl3_18
dynamic_agent_dim_N1_fixed_k_plugin_fp16 | 20 | 0.0174 | BEV backbone Conv |  | /layer0/layer0.0/conv2/Conv
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 1 | 0.7383 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.0/Conv + /shrink_conv/layers.0/double_conv/double_conv.1/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 2 | 0.4946 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.2/Conv + /shrink_conv/layers.0/double_conv/double_conv.3/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 3 | 0.3860 | BEV backbone Conv |  | /layer0/layer0.1/conv2_1/Conv + /layer0/layer0.1/relu_3/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 4 | 0.2611 | Other |  | __myl_TranReshMulAddReshMoveTranReluMaxr_myl2_19
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 5 | 0.2427 | PillarVFE / PFN |  | /pillar_vfe/pfn_layers_0/linear/MatMul_myl2_18
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 6 | 0.1229 | BEV backbone Conv |  | /layer0/layer0.0/conv2_1/Conv + /layer0/layer0.0/relu_3/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 7 | 0.1229 | BEV backbone Conv |  | /layer0/layer0.2/conv2_1/Conv + /layer0/layer0.2/relu_3/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 8 | 0.1115 | BEV backbone Conv |  | /layer0/layer0.0/conv1/Conv + /layer0/layer0.0/relu/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 9 | 0.1004 | BEV backbone Conv |  | /layer1/layer1.0/conv2/Conv + /layer1/layer1.0/relu_1/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 10 | 0.0881 | PointPillarScatterTRT |  | /PointPillarScatterTRT
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 11 | 0.0778 | BEV backbone Conv |  | /layer1/layer1.0/conv1/Conv + /layer1/layer1.0/relu/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 12 | 0.0778 | BEV backbone Conv |  | /layer0/layer0.2/conv2/Conv + /layer0/layer0.2/Add + /layer0/layer0.2/relu_1/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 13 | 0.0768 | BEV backbone Conv |  | /layer0/layer0.0/conv2/Conv
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 14 | 0.0768 | BEV backbone Conv |  | /layer0/layer0.1/conv2/Conv + /layer0/layer0.1/Add + /layer0/layer0.1/relu_1/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 15 | 0.0758 | BEV backbone Conv |  | /layer0/layer0.1/conv1/Conv + /layer0/layer0.1/relu/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 16 | 0.0757 | BEV backbone Conv |  | /layer0/layer0.2/conv1/Conv + /layer0/layer0.2/relu/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 17 | 0.0637 | BEV backbone Conv |  | /layer1/layer1.3/conv2/Conv + /layer1/layer1.3/relu_1/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 18 | 0.0636 | BEV backbone Conv |  | /layer1/layer1.4/conv2/Conv + /layer1/layer1.4/relu_1/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 19 | 0.0636 | BEV backbone Conv |  | /layer1/layer1.2/conv2/Conv + /layer1/layer1.2/relu_1/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp32 | 20 | 0.0635 | BEV backbone Conv |  | /layer1/layer1.1/conv2/Conv + /layer1/layer1.1/relu_1/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 1 | 0.2222 | Other |  | __myl_TranReshMulAddReshMoveTranReluMaxr_myl3_20
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 2 | 0.1884 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.0/Conv + /shrink_conv/layers.0/double_conv/double_conv.1/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 3 | 0.1270 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.2/Conv + /shrink_conv/layers.0/double_conv/double_conv.3/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 4 | 0.1065 | PillarVFE / PFN |  | /pillar_vfe/pfn_layers_0/linear/MatMul_myl3_19
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 5 | 0.0727 | PointPillarScatterTRT |  | /PointPillarScatterTRT
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 6 | 0.0553 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_GridCastNegExpAddDivAddGridCastEqlReplSeleSlicSlicMaxSubExpSlicSlicAddDivMulIsnaReplSeleMulEtc_myl67_10
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 7 | 0.0481 | BEV backbone Conv |  | /layer0/layer0.2/conv2_1/Conv + /layer0/layer0.2/relu_3/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 8 | 0.0471 | BEV backbone Conv |  | /layer0/layer0.0/conv2_1/Conv + /layer0/layer0.0/relu_3/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 9 | 0.0471 | BEV backbone Conv |  | /layer0/layer0.1/conv2_1/Conv + /layer0/layer0.1/relu_3/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 10 | 0.0348 | PointPillarScatterTRT |  | Reformatting CopyNode for Output Tensor 0 to /PointPillarScatterTRT
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 11 | 0.0299 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_GridCastNegExpAddDivAddGridCastEqlReplSeleSlicSlicMaxSubExpSlicSlicAddDivMulIsnaReplSeleMulEtc_myl67_8
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 12 | 0.0297 | BEV backbone Conv |  | /layer1/layer1.0/conv2/Conv + /layer1/layer1.0/relu_1/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 13 | 0.0287 | Other |  | __myl_Repl_myl3_4
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 14 | 0.0287 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_ReplReshReplReshReplReshIotaCastReshReplReshIotaCastReshReplReshConcCastConcCastConcCast_myl3_6
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 15 | 0.0287 | BEV backbone Conv |  | /layer0/layer0.1/conv2/Conv + /layer0/layer0.1/Add + /layer0/layer0.1/relu_1/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 16 | 0.0284 | BEV backbone Conv |  | /layer0/layer0.0/conv2/Conv
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 17 | 0.0278 | BEV backbone Conv |  | /layer1/layer1.3/conv2/Conv + /layer1/layer1.3/relu_1/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 18 | 0.0276 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_GtrReplSeleReshCastReshReshGtrReshCastDivMulSubConcMul_myl3_17
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 19 | 0.0276 | BEV backbone Conv |  | /layer0/layer0.0/conv1/Conv + /layer0/layer0.0/relu/Relu
dynamic_agent_dim_N2_fixed_k_plugin_fp16 | 20 | 0.0276 | BEV backbone Conv |  | /layer1/layer1.1/conv2/Conv + /layer1/layer1.1/relu_1/Relu
