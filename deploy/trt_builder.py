"""TensorRT engine builder wrapping trtexec.

Generates trtexec commands with dynamic shape profiles and plugin library
configuration, executes the build, and parses layer-level profiling output.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from ..utils.io_utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


class TRTBuilder:
    """Wraps trtexec for building TensorRT engines from ONNX models.

    Generates the full trtexec command with:
    - Dynamic shape profiles (--minShapes/--optShapes/--maxShapes)
    - Plugin library (--staticPlugins)
    - FP16 mode
    - Verbose profiling

    Args:
        trt_root: Path to TensorRT installation root (optional).
        plugin_so: Path to the HEAL TRT plugin shared library.
        agent_profile: Dict with 'min', 'opt', 'max' agent counts.
        trtexec_path: Path to trtexec binary (default: 'trtexec').
    """

    def __init__(
        self,
        trt_root: str | None = None,
        plugin_so: str | None = None,
        agent_profile: dict[str, int] | None = None,
        trtexec_path: str = "trtexec",
    ):
        self.trt_root = trt_root
        self.plugin_so = plugin_so
        self.agent_profile = agent_profile or {"min": 1, "opt": 2, "max": 2}
        self.trtexec_path = trtexec_path
        if trt_root:
            candidate = str(Path(trt_root) / "bin" / "trtexec")
            if Path(candidate).exists():
                self.trtexec_path = candidate

    def build_command(
        self,
        onnx_path: str,
        engine_path: str,
        input_shapes: dict[str, list[int]] | None = None,
        num_cams: int = 4,
        image_h: int = 256,
        image_w: int = 352,
    ) -> str:
        """Generate the trtexec command string.

        Args:
            onnx_path: Path to the ONNX model.
            engine_path: Path for the output .engine file.
            input_shapes: Optional explicit input shapes override.
            num_cams: Number of cameras per agent.
            image_h: Image height.
            image_w: Image width.

        Returns:
            Complete trtexec command string.
        """
        n_min = self.agent_profile["min"]
        n_opt = self.agent_profile["opt"]
        n_max = self.agent_profile["max"]

        if input_shapes is None:
            input_shapes = self._default_shapes(
                n_min, n_opt, n_max, num_cams, image_h, image_w
            )

        args = [
            self.trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16",
        ]

        if self.plugin_so:
            args.append(f"--staticPlugins={self.plugin_so}")

        # Dynamic shapes
        min_shapes = []
        opt_shapes = []
        max_shapes = []

        for name, (s_min, s_opt, s_max) in self._build_shape_profiles(
            n_min, n_opt, n_max, num_cams, image_h, image_w
        ).items():
            min_shapes.append(f"{name}:{s_min}")
            opt_shapes.append(f"{name}:{s_opt}")
            max_shapes.append(f"{name}:{s_max}")

        if min_shapes:
            args.append(f"--minShapes={','.join(min_shapes)}")
            args.append(f"--optShapes={','.join(opt_shapes)}")
            args.append(f"--maxShapes={','.join(max_shapes)}")

        args.extend(["--verbose", "--profilingVerbosity=detailed"])

        return " \\\n  ".join(args)

    def build(
        self,
        onnx_path: str,
        output_dir: str,
        engine_name: str = "model.engine",
        num_cams: int = 4,
        image_h: int = 256,
        image_w: int = 352,
        timeout: int = 600,
    ) -> dict[str, Any]:
        """Execute trtexec and build the TRT engine.

        Args:
            onnx_path: Path to the ONNX model.
            output_dir: Output directory for engine and logs.
            engine_name: Engine filename.
            num_cams: Number of cameras.
            image_h: Image height.
            image_w: Image width.
            timeout: Build timeout in seconds.

        Returns:
            Dict with 'engine_path', 'command', 'success', and 'profile'.
        """
        out = ensure_dir(output_dir)
        engine_path = str(Path(out) / engine_name)

        cmd = self.build_command(
            onnx_path, engine_path,
            num_cams=num_cams, image_h=image_h, image_w=image_w,
        )

        logger.info(f"Building TRT engine:\n{cmd}")

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout,
            )
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired:
            success = False
            stdout = ""
            stderr = f"trtexec timed out after {timeout}s"
        except Exception as exc:
            success = False
            stdout = ""
            stderr = str(exc)

        # Parse layer profile from stdout
        profile = self._parse_layer_profile(stdout) if success else {}

        # Save outputs
        log_path = str(Path(out) / "trtexec.log")
        with open(log_path, "w") as f:
            f.write(f"=== COMMAND ===\n{cmd}\n\n")
            f.write(f"=== STDOUT ===\n{stdout}\n\n")
            f.write(f"=== STDERR ===\n{stderr}\n")

        if profile:
            save_json(profile, str(Path(out) / "trt_layer_profile.json"))

        build_result = {
            "engine_path": engine_path if success else None,
            "command": cmd,
            "success": success,
            "log": log_path,
            "profile": profile,
        }

        save_json(build_result, str(Path(out) / "trt_build_result.json"))
        return build_result

    def _build_shape_profiles(
        self, n_min: int, n_opt: int, n_max: int,
        num_cams: int, h: int, w: int,
    ) -> dict[str, tuple[str, str, str]]:
        """Build min/opt/max shape profile strings.

        Args:
            n_min/n_opt/n_max: Agent count range.
            num_cams: Cameras per agent.
            h: Image height.
            w: Image width.

        Returns:
            Map of input_name -> (min_shape, opt_shape, max_shape).
        """
        def _shape(n: int, *dims: int) -> str:
            return "x".join(str(d) for d in (n, *dims))

        return {
            "imgs": (
                _shape(n_min, num_cams, 3, h, w),
                _shape(n_opt, num_cams, 3, h, w),
                _shape(n_max, num_cams, 3, h, w),
            ),
            "rots": (
                _shape(n_min, num_cams, 3, 3),
                _shape(n_opt, num_cams, 3, 3),
                _shape(n_max, num_cams, 3, 3),
            ),
            "trans": (
                _shape(n_min, num_cams, 3),
                _shape(n_opt, num_cams, 3),
                _shape(n_max, num_cams, 3),
            ),
            "intrins": (
                _shape(n_min, num_cams, 3, 3),
                _shape(n_opt, num_cams, 3, 3),
                _shape(n_max, num_cams, 3, 3),
            ),
            "post_rots": (
                _shape(n_min, num_cams, 3, 3),
                _shape(n_opt, num_cams, 3, 3),
                _shape(n_max, num_cams, 3, 3),
            ),
            "post_trans": (
                _shape(n_min, num_cams, 3),
                _shape(n_opt, num_cams, 3),
                _shape(n_max, num_cams, 3),
            ),
            "pairwise_t_matrix": (
                f"1x{n_min}x{n_min}x4x4",
                f"1x{n_opt}x{n_opt}x4x4",
                f"1x{n_max}x{n_max}x4x4",
            ),
        }

    def _default_shapes(
        self, n_min: int, n_opt: int, n_max: int,
        num_cams: int, h: int, w: int,
    ) -> dict[str, list[int]]:
        """Build default input shapes for opt profile."""
        return {
            "imgs": [n_opt, num_cams, 3, h, w],
            "rots": [n_opt, num_cams, 3, 3],
            "trans": [n_opt, num_cams, 3],
            "intrins": [n_opt, num_cams, 3, 3],
            "post_rots": [n_opt, num_cams, 3, 3],
            "post_trans": [n_opt, num_cams, 3],
            "pairwise_t_matrix": [1, n_opt, n_opt, 4, 4],
        }

    @staticmethod
    def _parse_layer_profile(stdout: str) -> dict[str, Any]:
        """Parse trtexec verbose output for layer-level profiling.

        Args:
            stdout: trtexec stdout text.

        Returns:
            Dict with layer names, precisions, and timings.
        """
        layers = []
        for line in stdout.split("\n"):
            if "Layer(" in line or "layer(" in line:
                layers.append(line.strip())
        return {"raw_layers": layers, "num_layers": len(layers)}
