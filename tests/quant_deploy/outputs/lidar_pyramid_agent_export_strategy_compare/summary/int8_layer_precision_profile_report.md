# INT8 Layer Precision Profile

- INT8 Conv layers count: 355
- FP16 Conv layers count: 2
- FP32 Conv layers count: 0
- PointPillarScatterTRT precision: ['fp16']
- Reformat/Cast/Q/DQ layers: 85

rank | latency_ms | precision | category | layer_type | layer
--- | --- | --- | --- | --- | ---
1 | 0.0 | unknown | reformat_cast_qdq | NoOp | Reformatting CopyNode for Network Input voxel_coords
2 | 0.0 | fp32 | reformat_cast_qdq | NoOp | Reformatting CopyNode for Network Input valid_voxel_mask
3 | 0.0 | fp16 | reformat_cast_qdq | Reformat | Reformatting CopyNode for Input Tensor 0 to {ForeignNode[ONNXTRT_ShapeShuffle_342_output[Constant].../pillar_vfe/Squeeze]}
4 | 0.0 | fp16 | reformat_cast_qdq | Reformat | Reformatting CopyNode for Input Tensor 3 to {ForeignNode[ONNXTRT_ShapeShuffle_342_output[Constant].../pillar_vfe/Squeeze]}
5 | 0.0 | unknown | other | signal | entry^bb^signal^1_myl4_0
6 | 0.0 | unknown | other | wait | entry^bb^wait^2_myl4_1
7 | 0.0 | unknown | other | wait | entry^bb^wait^1_myl4_2
8 | 0.0 | fp16 | other | kgen | __myl_Repl_myl4_3
9 | 0.0 | unknown | reformat_cast_qdq | kgen | __myl_ReplReshReplReshReplReshIotaCastReshReplReshIotaCastReshReplReshConcCastConcCastConcCast_myl4_4
10 | 0.0 | unknown | other | signal | __mye31311_myl4_5
11 | 0.0 | fp16 | reformat_cast_qdq | kgen | __myl_MoveMoveSlicReshCastReshSlicReshSlicReshCastReshSlicReshCastReshSlicReshSlicReshMulMulMulEtc_myl4_6
12 | 0.0 | unknown | other | wait | __mye31313_myl4_7
13 | 0.0 | fp16 | other | kgen | __myl_Scat_myl4_8
14 | 0.0 | fp16 | other | kgen | __myl_Scat_myl4_9
15 | 0.0 | fp16 | other | kgen | __myl_Scat_myl4_10
16 | 0.0 | fp16 | other | kgen | __myl_SlicSum_myl4_11
17 | 0.0 | unknown | other | signal | __mye31315_myl4_12
18 | 0.0 | unknown | other | wait | __mye31317_myl4_13
19 | 0.0 | fp16 | reformat_cast_qdq | kgen | __myl_GtrReplSeleReshCastReshReshGtrReshCastDivMulSubConcMul_myl4_14
20 | 0.0 | fp16 | PillarVFE/PFN | gemm | /pillar_vfe/pfn_layers_0/linear/MatMul_myl4_15
