"""CUDA PointPillar frontend shared by DAIR and CARLA post-scatter engines."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch


class ExternalPointPillarFrontend:
    """Run candidate PFN/scatter weights outside TensorRT without a K ceiling."""

    def __init__(
        self,
        hypes: Mapping[str, Any],
        checkpoint: str | Path,
        device: torch.device,
        *,
        modality: str = "m1",
    ) -> None:
        from opencood.models.heter_encoders import PointPillar

        self.modality = str(modality)
        encoder_config = copy.deepcopy(
            hypes["model"]["args"][self.modality]["encoder_args"]
        )
        self.encoder = PointPillar(encoder_config)
        payload = torch.load(Path(checkpoint), map_location="cpu")
        state_dict = payload.get("model", payload)
        if not isinstance(state_dict, Mapping):
            raise RuntimeError("post_scatter_frontend_checkpoint_state_missing")
        prefix = f"encoder_{self.modality}."
        encoder_state = {
            str(name)[len(prefix) :]: value
            for name, value in state_dict.items()
            if str(name).startswith(prefix)
        }
        if not encoder_state:
            raise RuntimeError(
                f"post_scatter_frontend_checkpoint_encoder_missing:{prefix}"
            )
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.encoder.to(device).eval()
        self.device = device

    def encode(self, ego_batch: Mapping[str, Any]) -> tuple[torch.Tensor, float]:
        source = ego_batch[f"inputs_{self.modality}"]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            encoded = self.encoder.pillar_vfe(source)
            spatial = self.encoder.scatter(encoded)["spatial_features"]
        end.record()
        end.synchronize()
        record_len = int(ego_batch["record_len"][0].item())
        if int(spatial.shape[0]) != record_len:
            raise RuntimeError(
                f"post_scatter_frontend_agent_mismatch:{int(spatial.shape[0])}!={record_len}"
            )
        return spatial.float().contiguous(), float(start.elapsed_time(end))

    def engine_inputs(
        self,
        ego_batch: Mapping[str, Any],
        spatial: torch.Tensor,
        *,
        input_names: set[str],
        max_agents: int = 2,
    ) -> dict[str, torch.Tensor]:
        record_len = int(ego_batch["record_len"][0].item())
        pairwise_source = ego_batch["pairwise_t_matrix"].float()
        if "agent_mask" not in input_names:
            result = {
                "spatial_features": spatial,
                "pairwise_t_matrix": pairwise_source[
                    :, :record_len, :record_len
                ].contiguous(),
            }
        else:
            if record_len > int(max_agents):
                raise RuntimeError(
                    f"post_scatter_frontend_agents_exceed_profile:{record_len}:{max_agents}"
                )
            if record_len < int(max_agents):
                spatial = torch.cat(
                    (
                        spatial,
                        spatial.new_zeros(
                            (int(max_agents) - record_len, *spatial.shape[1:])
                        ),
                    ),
                    dim=0,
                )
            pairwise = torch.eye(
                4, dtype=pairwise_source.dtype, device=pairwise_source.device
            ).reshape(1, 1, 1, 4, 4).repeat(
                1, int(max_agents), int(max_agents), 1, 1
            )
            pairwise[:, :record_len, :record_len] = pairwise_source[
                :, :record_len, :record_len
            ]
            mask = spatial.new_zeros((1, int(max_agents)))
            mask[:, :record_len] = 1.0
            result = {
                "spatial_features": spatial.contiguous(),
                "pairwise_t_matrix": pairwise.contiguous(),
                "agent_mask": mask.contiguous(),
            }
        if set(result) != set(input_names):
            raise RuntimeError(
                f"post_scatter_engine_input_mismatch:{sorted(result)}!={sorted(input_names)}"
            )
        return result


__all__ = ["ExternalPointPillarFrontend"]
