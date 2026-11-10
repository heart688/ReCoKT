"""Parameter-only cores for the school-withdrawal baseline runner.

This module includes an output-layer negation utility and FedEraser calibration.
The paper's NoT baseline is implemented by ``baseline_training.recover``.
FedEraser implements the paper's per-client Eq. (1),
and the caller supplies training data and recovery steps.
"""

from collections.abc import Mapping
import math

import torch
from torch import nn

__all__ = ["negate_output_layer", "calibrate_delta"]


def _check_tensor(value, label):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.layout != torch.strided or not value.is_floating_point():
        raise TypeError(f"{label} must be a dense real floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite tensor: {label}")


@torch.no_grad()
def negate_output_layer(model: nn.Module) -> nn.Module:
    """Negate all rows of ``model.head.weight`` and its optional bias in place.

    The head must be an explicit ``nn.Linear`` (DKT's output layer). For a fixed
    hidden state h, (-W)h + (-b) = -(Wh + b); for binary KT, sigmoid(-z) =
    1 - sigmoid(z). This standalone utility does not run the paper's NoT
    baseline. For that baseline, ``baseline_training.recover`` negates the
    first named parameter tensor and then trains on the remaining schools.
    Return the same model, preserving Parameter identities, requires_grad and
    train/eval mode.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    head = getattr(model, "head", None)
    if not isinstance(head, nn.Linear):
        raise ValueError("Expected an explicit nn.Linear output layer at model.head")
    # Validate BOTH parameters before changing either: failure is atomic.
    parameters = [("head.weight", head.weight)]
    if head.bias is not None:
        parameters.append(("head.bias", head.bias))
    for name, parameter in parameters:
        _check_tensor(parameter, name)
    for _, parameter in parameters:
        parameter.neg_()
    return model


def _l2_parts(delta):
    """Return scale and norm(delta / scale), avoiding squared-norm overflow.

    Accumulate on CPU in float64, including for MPS which lacks float64.
    Empty tensors contribute zero. No epsilon discards small nonzero updates.
    """
    scale = max((float(value.detach().abs().max()) if value.numel() else 0.0)
                for value in delta.values())
    if scale == 0.0:
        return 0.0, 0.0
    squared = math.fsum(float((value.detach().to(device="cpu", dtype=torch.float64)
                              / scale).square().sum())
                        for value in delta.values())
    return scale, math.sqrt(squared)


@torch.no_grad()
def calibrate_delta(old: Mapping[str, torch.Tensor],
                    fresh: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """FedEraser Eq. (1): ||old||_2 * fresh / ||fresh||_2, per client/round.

    Both mappings must contain the same complete floating parameter delta:
    local_after - broadcast_before, NOT local model weights or gradients.
    Flattening is conceptual: the L2 norm includes EVERY tensor together.
    Calibrate each retained client before weighted aggregation, then add the
    result to the current reconstructed model. The runner reuses original R
    deltas for round zero from the identical clean initialization.

    Return detached, independently allocated tensors with fresh's key order,
    dtypes and devices; leave both inputs untouched. Old/fresh devices may
    differ, but corresponding shapes/dtypes must agree. Integer state buffers
    need a separate runner policy and are rejected. Zero old -> zero output;
    nonzero old with zero fresh -> error (no invented replacement direction).
    """
    if not isinstance(old, Mapping) or not isinstance(fresh, Mapping):
        raise TypeError("old and fresh must be parameter-delta mappings")
    if not old or old.keys() != fresh.keys():
        raise ValueError("Nonempty old and fresh deltas must have identical keys")
    for name in old:
        if not isinstance(name, str):
            raise TypeError("Parameter names must be strings")
        _check_tensor(old[name], f"old[{name}]")
        _check_tensor(fresh[name], f"fresh[{name}]")
        if old[name].shape != fresh[name].shape:
            raise ValueError(f"Delta shape mismatch: {name}")
        if old[name].dtype != fresh[name].dtype:
            raise TypeError(f"Delta dtype mismatch: {name}")

    old_scale, old_length = _l2_parts(old)
    if old_scale == 0.0:
        return {name: torch.zeros_like(value) for name, value in fresh.items()}
    fresh_scale, fresh_length = _l2_parts(fresh)
    if fresh_scale == 0.0:
        raise FloatingPointError("Zero fresh direction with nonzero historical delta")

    result = {}
    for name, value in fresh.items():
        # Normalize BEFORE restoring magnitude: old_norm/fresh_norm could
        # overflow even when every final calibrated coordinate is finite.
        direction = value.detach().to(device="cpu", dtype=torch.float64) / fresh_scale
        calibrated = ((direction / fresh_length) * old_length) * old_scale
        calibrated = calibrated.to(device=value.device, dtype=value.dtype)
        if not bool(torch.isfinite(calibrated).all()):
            raise FloatingPointError(f"Nonfinite calibrated delta: {name}")
        result[name] = calibrated
    return result
