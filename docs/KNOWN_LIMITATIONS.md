# Known limitations

- Transformer and attention channel pruning has not been validated and is not
  presented as a supported production capability.
- Dynamic Python control flow that cannot be represented or resolved by the
  tracer requires an adapter or an explicitly supported trace. Default behavior
  is fail closed.
- The HEAL signal-maxK exporter still requires the caller's HEAL model adapter,
  checkpoint/config and representative inputs. This formalization did not run
  an export.
- TensorRT engine construction/runtime requires an external TensorRT 10.x
  installation and, for PointPillar scatter, a compatible plugin. The package
  contains the interfaces but this formalization did not build or run an
  engine.
- Only FP16 plus INT8 explicit Q/DQ deployment is supported. INT4, arbitrary
  bit width and weight-only quantization are not implemented.
- Detection metric helpers cannot infer a project-specific decoder from raw
  tensors; callers must provide decoded predictions or an explicit decoder.
- Legacy grouped-convolution artifacts without `group_keep_map` cannot be
  replayed under `independent_group_topk`.

