"""Exceptions raised by the formal FP16/INT8 deployment pipeline."""

from __future__ import annotations


class QuantizationError(RuntimeError):
    """Base class for formal quantization and deployment errors."""


class ArtifactSchemaError(QuantizationError):
    """An artifact does not satisfy its declared schema."""


class OnnxExportError(QuantizationError):
    """Signal-maxK ONNX export or export validation failed."""


class CanonicalMappingError(QuantizationError):
    """A weighted PyTorch/ONNX mapping cannot be proven uniquely."""


class AmbiguousCanonicalMappingError(CanonicalMappingError):
    """More than one canonical mapping is possible."""


class QDQInsertionError(QuantizationError):
    """Explicit Q/DQ insertion could not be completed safely."""


class QDQValidationError(QuantizationError):
    """A Q/DQ graph cannot be traced to its physical weights."""


class PhysicalStructureMismatchError(QDQValidationError):
    """ONNX/QDQ weights disagree with the physical model snapshot."""


class TensorRTConfigurationError(QuantizationError):
    """TensorRT command configuration is incomplete or inconsistent."""


class TensorRTBuildError(QuantizationError):
    """An explicitly requested TensorRT build failed."""


class EngineStructureError(QuantizationError):
    """TensorRT engine structure is inconsistent with canonical metadata."""


class PrecisionRealizationError(QuantizationError):
    """Requested and realized TensorRT precisions disagree."""


class ProvenanceError(QuantizationError):
    """Deployment provenance is incomplete or inconsistent."""
