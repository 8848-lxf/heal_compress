# TensorRT Layer Bottleneck Report

profile | rank | latency_ms | category | layer_type | layer
--- | --- | --- | --- | --- | ---
wide_fp32 | 1 | 1.0199 | BEV backbone Conv |  | /layer0/layer0.2/conv2/Conv + /layer0/layer0.2/Add + /layer0/layer0.2/relu_1/Relu
wide_fp32 | 2 | 0.7496 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.0/Conv + /shrink_conv/layers.0/double_conv/double_conv.1/Relu
wide_fp32 | 3 | 0.4997 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.2/Conv + /shrink_conv/layers.0/double_conv/double_conv.3/Relu
wide_fp32 | 4 | 0.2796 | Other |  | __myl_TranReshMulAddReshMoveTranReluMaxr_myl0_23
wide_fp32 | 5 | 0.2243 | PillarVFE / PFN |  | /pillar_vfe/pfn_layers_0/linear/MatMul_myl0_20
wide_fp32 | 6 | 0.1997 | Other |  | __myl_Move_myl0_21
wide_fp32 | 7 | 0.1976 | Other |  | __myl_Repl_myl0_6
wide_fp32 | 8 | 0.1331 | Other |  | __myl_Mul_myl0_28
wide_fp32 | 9 | 0.1229 | BEV backbone Conv |  | /layer0/layer0.0/conv2_1/Conv + /layer0/layer0.0/relu_3/Relu
wide_fp32 | 10 | 0.1229 | BEV backbone Conv |  | /layer0/layer0.2/conv2_1/Conv + /layer0/layer0.2/relu_3/Relu
wide_fp32 | 11 | 0.1219 | BEV backbone Conv |  | /layer0/layer0.1/conv2_1/Conv + /layer0/layer0.1/relu_3/Relu
wide_fp32 | 12 | 0.1208 | BEV backbone Conv |  | /layer0/layer0.0/conv1/Conv + /layer0/layer0.0/relu/Relu
wide_fp32 | 13 | 0.1065 | Other |  | __myl_ReshTran_myl0_27
wide_fp32 | 14 | 0.0952 | BEV backbone Conv |  | /layer1/layer1.0/conv2/Conv + /layer1/layer1.0/relu_1/Relu
wide_fp32 | 15 | 0.0922 | BEV backbone Conv |  | /deblocks.2/deblocks.2.0/ConvTranspose + /deblocks.2/deblocks.2.1/BatchNormalization + /deblocks.2/deblocks.2.2/Relu
wide_fp32 | 16 | 0.0778 | BEV backbone Conv |  | /layer1/layer1.0/conv1/Conv + /layer1/layer1.0/relu/Relu
wide_fp32 | 17 | 0.0768 | BEV backbone Conv |  | /layer0/layer0.1/conv2/Conv + /layer0/layer0.1/Add + /layer0/layer0.1/relu_1/Relu
wide_fp32 | 18 | 0.0758 | BEV backbone Conv |  | /layer0/layer0.1/conv1/Conv + /layer0/layer0.1/relu/Relu
wide_fp32 | 19 | 0.0748 | BEV backbone Conv |  | /layer0/layer0.0/conv2/Conv
wide_fp32 | 20 | 0.0737 | BEV backbone Conv |  | /layer0/layer0.2/conv1/Conv + /layer0/layer0.2/relu/Relu
wide_fp16 | 1 | 0.2365 | Other |  | __myl_TranReshMulAddReshMoveTranReluMaxr_myl2_24
wide_fp16 | 2 | 0.1905 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.0/Conv + /shrink_conv/layers.0/double_conv/double_conv.1/Relu
wide_fp16 | 3 | 0.1288 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.2/Conv + /shrink_conv/layers.0/double_conv/double_conv.3/Relu
wide_fp16 | 4 | 0.1126 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_CastReshMulMulMulNegExpAddDivAddMulNegExpAddDivAddMulNegExpAddDivAddMulGridCastGridCastGridEtc_myl65_8
wide_fp16 | 5 | 0.1056 | PillarVFE / PFN |  | /pillar_vfe/pfn_layers_0/linear/MatMul_myl2_21
wide_fp16 | 6 | 0.1034 | Other |  | __myl_Move_myl2_22
wide_fp16 | 7 | 0.0993 | Other |  | __myl_Repl_myl2_6
wide_fp16 | 8 | 0.0532 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | Reformatting CopyNode for Output Tensor 0 to {ForeignNode[ONNXTRT_ShapeShuffle_336_output[Constant].../Mul_2]}
wide_fp16 | 9 | 0.0481 | BEV backbone Conv |  | /layer0/layer0.2/conv2_1/Conv + /layer0/layer0.2/relu_3/Relu
wide_fp16 | 10 | 0.0471 | BEV backbone Conv |  | /layer0/layer0.0/conv2_1/Conv + /layer0/layer0.0/relu_3/Relu
wide_fp16 | 11 | 0.0461 | BEV backbone Conv |  | /layer0/layer0.1/conv2_1/Conv + /layer0/layer0.1/relu_3/Relu
wide_fp16 | 12 | 0.0399 | Other |  | __myl_ReshTran_myl2_28
wide_fp16 | 13 | 0.0389 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_IotaReplReshReplReshReplReshIotaCastReshSlicReplReshCastReshReplReshConcCastSlicReplReshEtc_myl2_8
wide_fp16 | 14 | 0.0379 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_ReshGtrReshCastCastReshDivMulSubConcMul_myl2_19
wide_fp16 | 15 | 0.0348 | Other |  | __myl_SlicSum_myl2_16
wide_fp16 | 16 | 0.0297 | BEV backbone Conv |  | /layer0/layer0.0/conv2/Conv
wide_fp16 | 17 | 0.0294 | BEV backbone Conv |  | /layer1/layer1.0/conv2/Conv + /layer1/layer1.0/relu_1/Relu
wide_fp16 | 18 | 0.0287 | BEV backbone Conv |  | /layer0/layer0.1/conv2/Conv + /layer0/layer0.1/Add + /layer0/layer0.1/relu_1/Relu
wide_fp16 | 19 | 0.0287 | BEV backbone Conv |  | /layer1/layer1.1/conv2/Conv + /layer1/layer1.1/relu_1/Relu
wide_fp16 | 20 | 0.0287 | BEV backbone Conv |  | /layer1/layer1.2/conv2/Conv + /layer1/layer1.2/relu_1/Relu
bucketed_fp32 | 1 | 1.1525 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.2/Conv + /shrink_conv/layers.0/double_conv/double_conv.3/Relu
bucketed_fp32 | 2 | 0.9170 | BEV backbone Conv |  | /layer0/layer0.2/conv2/Conv + /layer0/layer0.2/Add + /layer0/layer0.2/relu_1/Relu
bucketed_fp32 | 3 | 0.9052 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.0/Conv + /shrink_conv/layers.0/double_conv/double_conv.1/Relu
bucketed_fp32 | 4 | 0.2725 | Other |  | __myl_TranReshMulAddReshMoveTranReluMaxr_myl0_23
bucketed_fp32 | 5 | 0.2386 | PillarVFE / PFN |  | /pillar_vfe/pfn_layers_0/linear/MatMul_myl0_20
bucketed_fp32 | 6 | 0.1966 | Other |  | __myl_Move_myl0_21
bucketed_fp32 | 7 | 0.1900 | Other |  | __myl_Repl_myl0_6
bucketed_fp32 | 8 | 0.1331 | Other |  | __myl_Mul_myl0_28
bucketed_fp32 | 9 | 0.1239 | BEV backbone Conv |  | /layer0/layer0.0/conv2_1/Conv + /layer0/layer0.0/relu_3/Relu
bucketed_fp32 | 10 | 0.1229 | BEV backbone Conv |  | /layer0/layer0.0/conv1/Conv + /layer0/layer0.0/relu/Relu
bucketed_fp32 | 11 | 0.1219 | BEV backbone Conv |  | /layer0/layer0.2/conv2_1/Conv + /layer0/layer0.2/relu_3/Relu
bucketed_fp32 | 12 | 0.1208 | BEV backbone Conv |  | /layer0/layer0.1/conv2_1/Conv + /layer0/layer0.1/relu_3/Relu
bucketed_fp32 | 13 | 0.1004 | Other |  | __myl_ReshTran_myl0_27
bucketed_fp32 | 14 | 0.0943 | BEV backbone Conv |  | /layer1/layer1.0/conv2/Conv + /layer1/layer1.0/relu_1/Relu
bucketed_fp32 | 15 | 0.0911 | BEV backbone Conv |  | /deblocks.2/deblocks.2.0/ConvTranspose + /deblocks.2/deblocks.2.1/BatchNormalization + /deblocks.2/deblocks.2.2/Relu
bucketed_fp32 | 16 | 0.0776 | BEV backbone Conv |  | /layer0/layer0.1/conv2/Conv + /layer0/layer0.1/Add + /layer0/layer0.1/relu_1/Relu
bucketed_fp32 | 17 | 0.0768 | BEV backbone Conv |  | /layer1/layer1.0/conv1/Conv + /layer1/layer1.0/relu/Relu
bucketed_fp32 | 18 | 0.0748 | BEV backbone Conv |  | /layer0/layer0.0/conv2/Conv
bucketed_fp32 | 19 | 0.0748 | BEV backbone Conv |  | /layer0/layer0.2/conv1/Conv + /layer0/layer0.2/relu/Relu
bucketed_fp32 | 20 | 0.0737 | BEV backbone Conv |  | /layer0/layer0.1/conv1/Conv + /layer0/layer0.1/relu/Relu
bucketed_fp16 | 1 | 0.2345 | Other |  | __myl_TranReshMulAddReshMoveTranReluMaxr_myl0_20
bucketed_fp16 | 2 | 0.1905 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.0/Conv + /shrink_conv/layers.0/double_conv/double_conv.1/Relu
bucketed_fp16 | 3 | 0.1382 | Other |  | __myl_Move_myl0_18
bucketed_fp16 | 4 | 0.1372 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_IotaReplReshReplReshReplReshIotaCastReshReplReshCastReshReplReshConcCastConcCastConcCast_myl0_7
bucketed_fp16 | 5 | 0.1362 | Other |  | __myl_Repl_myl0_5
bucketed_fp16 | 6 | 0.1281 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.2/Conv + /shrink_conv/layers.0/double_conv/double_conv.3/Relu
bucketed_fp16 | 7 | 0.1126 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_CastReshMulMulMulNegExpAddDivAddMulNegExpAddDivAddMulNegExpAddDivAddMulGridCastGridCastGridEtc_myl62_8
bucketed_fp16 | 8 | 0.1056 | PillarVFE / PFN |  | /pillar_vfe/pfn_layers_0/linear/MatMul_myl0_17
bucketed_fp16 | 9 | 0.0911 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_MoveSlicReshCastCastReshCastSlicReshSlicReshCastReshSlicReshCastCastReshSlicReshSlicReshEtc_myl0_9
bucketed_fp16 | 10 | 0.0489 | BEV backbone Conv |  | /layer0/layer0.2/conv2_1/Conv + /layer0/layer0.2/relu_3/Relu
bucketed_fp16 | 11 | 0.0473 | BEV backbone Conv |  | /layer0/layer0.0/conv2_1/Conv + /layer0/layer0.0/relu_3/Relu
bucketed_fp16 | 12 | 0.0461 | BEV backbone Conv |  | /layer0/layer0.1/conv2_1/Conv + /layer0/layer0.1/relu_3/Relu
bucketed_fp16 | 13 | 0.0410 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_ReshTranCastReshMul_myl0_24
bucketed_fp16 | 14 | 0.0328 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_ReshGtrReshCastCastReshDivMulSubConcMul_myl0_15
bucketed_fp16 | 15 | 0.0297 | BEV backbone Conv |  | /layer0/layer0.0/conv2/Conv
bucketed_fp16 | 16 | 0.0290 | BEV backbone Conv |  | /layer1/layer1.0/conv2/Conv + /layer1/layer1.0/relu_1/Relu
bucketed_fp16 | 17 | 0.0287 | BEV backbone Conv |  | /layer0/layer0.1/conv2/Conv + /layer0/layer0.1/Add + /layer0/layer0.1/relu_1/Relu
bucketed_fp16 | 18 | 0.0287 | BEV backbone Conv |  | /layer1/layer1.1/conv2/Conv + /layer1/layer1.1/relu_1/Relu
bucketed_fp16 | 19 | 0.0287 | BEV backbone Conv |  | /layer1/layer1.2/conv2/Conv + /layer1/layer1.2/relu_1/Relu
bucketed_fp16 | 20 | 0.0276 | BEV backbone Conv |  | /layer0/layer0.0/conv1/Conv + /layer0/layer0.0/relu/Relu
fixed_shape_fp32 | 1 | 1.0248 | BEV backbone Conv |  | /layer0/layer0.2/conv2/Conv + /layer0/layer0.2/Add + /layer0/layer0.2/relu_1/Relu
fixed_shape_fp32 | 2 | 0.7147 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.0/Conv + /shrink_conv/layers.0/double_conv/double_conv.1/Relu
fixed_shape_fp32 | 3 | 0.4833 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.2/Conv + /shrink_conv/layers.0/double_conv/double_conv.3/Relu
fixed_shape_fp32 | 4 | 0.2693 | Other |  | __myl_TranReshMulAddReshMoveTranReluMaxr_myl0_22
fixed_shape_fp32 | 5 | 0.2244 | PillarVFE / PFN |  | /pillar_vfe/pfn_layers_0/linear/MatMul_myl0_19
fixed_shape_fp32 | 6 | 0.1976 | Other |  | __myl_Move_myl0_20
fixed_shape_fp32 | 7 | 0.1946 | Other |  | __myl_Repl_myl0_5
fixed_shape_fp32 | 8 | 0.1331 | Other |  | __myl_Mul_myl0_27
fixed_shape_fp32 | 9 | 0.1239 | BEV backbone Conv |  | /layer0/layer0.0/conv2_1/Conv + /layer0/layer0.0/relu_3/Relu
fixed_shape_fp32 | 10 | 0.1231 | BEV backbone Conv |  | /layer0/layer0.2/conv2_1/Conv + /layer0/layer0.2/relu_3/Relu
fixed_shape_fp32 | 11 | 0.1229 | BEV backbone Conv |  | /layer0/layer0.1/conv2_1/Conv + /layer0/layer0.1/relu_3/Relu
fixed_shape_fp32 | 12 | 0.1208 | BEV backbone Conv |  | /layer0/layer0.0/conv1/Conv + /layer0/layer0.0/relu/Relu
fixed_shape_fp32 | 13 | 0.1004 | Other |  | __myl_ReshTran_myl0_26
fixed_shape_fp32 | 14 | 0.0952 | BEV backbone Conv |  | /layer1/layer1.0/conv2/Conv + /layer1/layer1.0/relu_1/Relu
fixed_shape_fp32 | 15 | 0.0901 | BEV backbone Conv |  | /deblocks.2/deblocks.2.0/ConvTranspose + /deblocks.2/deblocks.2.1/BatchNormalization + /deblocks.2/deblocks.2.2/Relu
fixed_shape_fp32 | 16 | 0.0781 | BEV backbone Conv |  | /layer1/layer1.0/conv1/Conv + /layer1/layer1.0/relu/Relu
fixed_shape_fp32 | 17 | 0.0768 | BEV backbone Conv |  | /layer0/layer0.1/conv2/Conv + /layer0/layer0.1/Add + /layer0/layer0.1/relu_1/Relu
fixed_shape_fp32 | 18 | 0.0749 | BEV backbone Conv |  | /layer0/layer0.2/conv1/Conv + /layer0/layer0.2/relu/Relu
fixed_shape_fp32 | 19 | 0.0748 | BEV backbone Conv |  | /layer0/layer0.0/conv2/Conv
fixed_shape_fp32 | 20 | 0.0748 | BEV backbone Conv |  | /layer0/layer0.1/conv1/Conv + /layer0/layer0.1/relu/Relu
fixed_shape_fp16 | 1 | 0.2222 | Other |  | __myl_TranReshMulAddReshMoveTranReluMaxr_myl0_19
fixed_shape_fp16 | 2 | 0.1905 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.0/Conv + /shrink_conv/layers.0/double_conv/double_conv.1/Relu
fixed_shape_fp16 | 3 | 0.1403 | Other |  | __myl_Move_myl0_17
fixed_shape_fp16 | 4 | 0.1362 | Other |  | __myl_Repl_myl0_4
fixed_shape_fp16 | 5 | 0.1352 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_IotaReplReshReplReshReplReshIotaCastReshReplReshCastReshReplReshConcCastConcCastConcCast_myl0_6
fixed_shape_fp16 | 6 | 0.1283 | BEV backbone Conv |  | /shrink_conv/layers.0/double_conv/double_conv.2/Conv + /shrink_conv/layers.0/double_conv/double_conv.3/Relu
fixed_shape_fp16 | 7 | 0.1126 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_CastReshMulMulMulNegExpAddDivAddMulNegExpAddDivAddMulNegExpAddDivAddMulGridCastGridCastGridEtc_myl62_8
fixed_shape_fp16 | 8 | 0.1096 | PillarVFE / PFN |  | /pillar_vfe/pfn_layers_0/linear/MatMul_myl0_16
fixed_shape_fp16 | 9 | 0.0901 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_MoveSlicReshCastCastReshCastSlicReshSlicReshCastReshSlicReshCastCastReshSlicReshSlicReshEtc_myl0_8
fixed_shape_fp16 | 10 | 0.0492 | BEV backbone Conv |  | /layer0/layer0.0/conv2_1/Conv + /layer0/layer0.0/relu_3/Relu
fixed_shape_fp16 | 11 | 0.0492 | BEV backbone Conv |  | /layer0/layer0.2/conv2_1/Conv + /layer0/layer0.2/relu_3/Relu
fixed_shape_fp16 | 12 | 0.0481 | BEV backbone Conv |  | /layer0/layer0.1/conv2_1/Conv + /layer0/layer0.1/relu_3/Relu
fixed_shape_fp16 | 13 | 0.0410 | Shape/Gather/Slice/Reshape/Shuffle/Reformat |  | __myl_ReshTranCastReshMul_myl0_23
fixed_shape_fp16 | 14 | 0.0297 | BEV backbone Conv |  | /layer0/layer0.0/conv2/Conv
fixed_shape_fp16 | 15 | 0.0297 | BEV backbone Conv |  | /layer1/layer1.0/conv2/Conv + /layer1/layer1.0/relu_1/Relu
fixed_shape_fp16 | 16 | 0.0287 | BEV backbone Conv |  | /layer0/layer0.1/conv2/Conv + /layer0/layer0.1/Add + /layer0/layer0.1/relu_1/Relu
fixed_shape_fp16 | 17 | 0.0287 | BEV backbone Conv |  | /layer1/layer1.1/conv2/Conv + /layer1/layer1.1/relu_1/Relu
fixed_shape_fp16 | 18 | 0.0287 | BEV backbone Conv |  | /layer1/layer1.2/conv2/Conv + /layer1/layer1.2/relu_1/Relu
fixed_shape_fp16 | 19 | 0.0286 | BEV backbone Conv |  | /layer1/layer1.4/conv2/Conv + /layer1/layer1.4/relu_1/Relu
fixed_shape_fp16 | 20 | 0.0277 | BEV backbone Conv |  | /layer1/layer1.3/conv2/Conv + /layer1/layer1.3/relu_1/Relu

category | total_ms
--- | ---
BEV backbone Conv | 20.1019
Other | 5.0877
Shape/Gather/Slice/Reshape/Shuffle/Reformat | 1.7418
PillarVFE / PFN | 1.0081
Pyramid fusion / BEV warp / GridSample | 0.3157
Head Conv | 0.2976
