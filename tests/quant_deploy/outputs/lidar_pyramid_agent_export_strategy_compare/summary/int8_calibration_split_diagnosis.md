# INT8 Calibration Split Diagnosis

- fixed_K: 29696
- old_dynamic_bucket_calibration_split_confirmed: false
- new_dynamic_bucket_calibration_split: train
- new_single_engine_calibration_split: train
- val_calib_diagnostic_executed: false

label | calibration_split | status | AP@0.70 | mAP | forward_p50 | note
--- | --- | --- | --- | --- | --- | ---
historical_dynamic_bucket_int8_calib200 | None | success | 0.5065 | 0.651 | 2.558410167694092 | historical only; split unconfirmed
new_dynamic_bucket_int8_train_calib200 | train | success | 0.4783 | 0.6194 | 2.7008913457393646 | 
new_single_engine_maxK_int8_train_calib200 | train | success | 0.4691 | 0.6124 | 2.8709396719932556 | 
new_dynamic_bucket_fp16_reference | None | success | 0.6013 | 0.7364 | 3.28262522816658 | 
new_single_engine_fp16_reference | None | success | 0.6012 | 0.7363 | 3.4408215433359146 | 

## Suspected Root Causes

- 1. historical calibration split or sample-selection mismatch: historical dynamic bucket INT8 mAP exceeds new train-calib dynamic bucket by 0.0316
- 2. activation quantization too aggressive for single-engine maxK: single-engine train-calib INT8 mAP drop vs FP16 is 0.1239
