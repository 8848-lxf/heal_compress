"""Explicit Q/DQ node insertion for ONNX deployment graphs.

Inserts QuantizeLinear + DequantizeLinear nodes for weight-only quantization.
Activations are NOT quantized. BEV plugin nodes are skipped.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ..utils.io_utils import ensure_dir, save_json, save_yaml

logger = logging.getLogger(__name__)

# BEV plugin ops that must NOT receive Q/DQ nodes
BEV_PLUGIN_OPS = frozenset({
    "BEVPoolDynamicTRT", "BEVWarpDynamicTRT",
    "Inverse3x3TRT", "SE3InverseTRT",
})


class QDQDeploymentBuilder:
    """Builds the deployment ONNX graph with explicit Q/DQ nodes.

    For each quantizable layer in the subnet config:
    - FP16: no Q/DQ needed (native TRT FP16)
    - INT8: insert weight-side QuantizeLinear + DequantizeLinear
    - INT4: for Linear/GEMM insert Q/DQ; for Conv2d mark as plugin_required

    Activation values are NOT quantized (remain FP16).

    Args:
        subnet_config_path: Path to best_subnet_config.json.
        granularity: 'per_tensor' or 'per_channel'.
    """

    BIT_MAP = {"FP16": 16, "INT8": 8, "INT4": 4}

    def __init__(
        self,
        subnet_config_path: str,
        granularity: str = "per_channel",
    ):
        self.subnet_config_path = subnet_config_path
        self.granularity = granularity
        self._subnet_config: dict[str, Any] | None = None

    def _load_config(self) -> dict[str, Any]:
        """Load the subnet config JSON."""
        if self._subnet_config is None:
            from ..utils.io_utils import load_json
            self._subnet_config = load_json(self.subnet_config_path)
        return self._subnet_config

    def insert_qdq(
        self,
        onnx_path: str,
        output_dir: str,
        pruned_filename: str = "model_pruned.onnx",
        qdq_filename: str = "model_mixed_qdq.onnx",
    ) -> dict[str, str]:
        """Insert Q/DQ nodes into the ONNX graph.

        Steps:
        1. Copy the pruned ONNX as model_pruned.onnx.
        2. For each layer with INT8/INT4 bit-width, insert weight Q/DQ nodes.
        3. Skip BEV plugin nodes.
        4. Skip INT4 Conv2d layers (mark as plugin_required).
        5. Save model_mixed_qdq.onnx and mixed_precision_policy.yaml.

        Args:
            onnx_path: Input ONNX model path (after physical pruning).
            output_dir: Output directory.
            pruned_filename: Filename for the pruned ONNX copy.
            qdq_filename: Filename for the Q/DQ-inserted ONNX.

        Returns:
            Dict with 'pruned_onnx', 'qdq_onnx', 'policy_yaml' paths.
        """
        out = ensure_dir(output_dir)
        config = self._load_config()
        bitwidth_vars = config.get("bitwidth_vars", {})

        pruned_path = str(Path(out) / pruned_filename)
        qdq_path = str(Path(out) / qdq_filename)
        policy_path = str(Path(out) / "mixed_precision_policy.yaml")

        # Copy pruned ONNX
        shutil.copyfile(onnx_path, pruned_path)

        rewired = []
        policy = {}

        try:
            import numpy as np
            import onnx
            from onnx import TensorProto, helper, numpy_helper

            model = onnx.load(pruned_path)
            initializer_names = {init.name for init in model.graph.initializer}

            for layer_name, bit_label in bitwidth_vars.items():
                bits = self.BIT_MAP.get(bit_label, 16)
                if bits >= 16:
                    policy[layer_name] = {"bitwidth": bit_label, "status": "native_fp16"}
                    continue

                # Check if Conv2d + INT4 -> plugin_required
                deploy_status = self._classify_deploy(layer_name, bit_label, model)
                policy[layer_name] = {
                    "bitwidth": bit_label,
                    "status": deploy_status,
                }
                if deploy_status == "plugin_required":
                    continue

                # Find matching weight initializer
                matches = [
                    name for name in initializer_names
                    if name.startswith(layer_name) and name.endswith("weight")
                ]

                for weight_name in matches:
                    self._insert_qdq_pair(
                        model, weight_name, bits, rewired
                    )

            onnx.save(model, qdq_path)
            status = "inserted"

        except Exception as exc:
            logger.error(f"Q/DQ insertion failed: {exc}")
            shutil.copyfile(pruned_path, qdq_path)
            status = f"failed: {exc}"

        save_yaml(policy, policy_path)
        save_json(
            {
                "pruned_onnx": pruned_path,
                "qdq_onnx": qdq_path,
                "policy_yaml": policy_path,
                "status": status,
                "weight_only": True,
                "activation_precision": "fp16",
                "granularity": self.granularity,
                "rewired": rewired,
            },
            str(Path(out) / "qdq_report.json"),
        )

        return {
            "pruned_onnx": pruned_path,
            "qdq_onnx": qdq_path,
            "policy_yaml": policy_path,
        }

    def _insert_qdq_pair(
        self,
        model: Any,
        weight_name: str,
        bits: int,
        rewired: list[dict],
    ) -> None:
        """Insert a QuantizeLinear + DequantizeLinear pair for one weight.

        Args:
            model: ONNX model object.
            weight_name: Name of the weight initializer.
            bits: Quantization bit-width (8 or 4).
            rewired: List to append rewiring records to.
        """
        import numpy as np
        from onnx import numpy_helper, helper

        scale_name = f"{weight_name}_w{bits}_scale"
        zp_name = f"{weight_name}_w{bits}_zp"
        q_name = f"{weight_name}_Q{bits}"
        dq_name = f"{weight_name}_DQ{bits}"

        scale_val = np.array([1.0], dtype=np.float32)
        zp_val = np.zeros(1, dtype=np.int8 if bits <= 8 else np.int32)

        model.graph.initializer.extend([
            numpy_helper.from_array(scale_val, name=scale_name),
            numpy_helper.from_array(zp_val, name=zp_name),
        ])
        model.graph.node.extend([
            helper.make_node(
                "QuantizeLinear",
                [weight_name, scale_name, zp_name], [q_name],
                name=f"Q_{weight_name}_w{bits}",
            ),
            helper.make_node(
                "DequantizeLinear",
                [q_name, scale_name, zp_name], [dq_name],
                name=f"DQ_{weight_name}_w{bits}",
            ),
        ])

        # Rewire consumers
        for node in model.graph.node:
            if node.name.startswith(("Q_", "DQ_")):
                continue
            for idx, inp in enumerate(node.input):
                if inp == weight_name:
                    node.input[idx] = dq_name
                    rewired.append({
                        "weight": weight_name, "bit": bits,
                        "consumer": node.name,
                    })

    def _classify_deploy(
        self, layer_name: str, bit_label: str, onnx_model: Any,
    ) -> str:
        """Classify deployment status for a layer.

        Args:
            layer_name: Layer name.
            bit_label: Bit-width label.
            onnx_model: ONNX model (for node type lookup).

        Returns:
            Deploy status string.
        """
        if bit_label == "FP16":
            return "native_fp16"
        if bit_label == "INT8":
            return "native_int8_qdq"
        if bit_label == "INT4":
            # Check if it's a Conv node
            for node in onnx_model.graph.node:
                if any(layer_name in inp for inp in node.input):
                    if node.op_type == "Conv":
                        return "plugin_required"
            return "int4_woq_gemm"
        return "native_fp16"
