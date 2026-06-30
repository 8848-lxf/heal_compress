# Padded Agent Static Train Calibration fixedK29696 calib200

- strategy: padded_agent_static
- calibration_split: train
- calibration_frames: 200
- fixed_K: 29696
- max_cav: 2
- npz_file_count: 200
- npz_dir: tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/train_calib_padded_agent_static_fixedK29696_200
- manifest: tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/train_calib_padded_agent_static_fixedK29696_200/manifest.json
- skipped_samples: 0
- record_len_distribution: {'1': 10, '2': 190}
- valid_agent_mask_distribution: {'[[1.0, 0.0]]': 10, '[[1.0, 1.0]]': 190}
