from __future__ import annotations

import torch
from torch import nn


def _make_ix_like(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    view = [1] * tensor.dim()
    view[dim] = -1
    return torch.arange(1, tensor.shape[dim] + 1, device=tensor.device, dtype=tensor.dtype).view(view)


def entmax15(inputs: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    if dim < 0:
        dim = inputs.dim() + dim

    sorted_inputs, _ = torch.sort(inputs, dim=dim, descending=True)
    rho = _make_ix_like(sorted_inputs, dim)
    mean = sorted_inputs.cumsum(dim=dim) / rho
    mean_sq = sorted_inputs.square().cumsum(dim=dim) / rho
    variance = rho * (mean_sq - mean.square())
    delta = torch.clamp((1.0 - variance) / rho, min=0.0)
    taus = mean - torch.sqrt(delta)
    support = (sorted_inputs >= taus).to(dtype=torch.int64)
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau_star = taus.gather(dim, support_size - 1)
    output = torch.clamp(inputs - tau_star, min=0.0).square()
    normalizer = output.sum(dim=dim, keepdim=True).clamp(min=eps)
    return output / normalizer


class Entmax15(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return entmax15(inputs, dim=self.dim)
