from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _make_ix_like(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    view = [1] * tensor.dim()
    view[dim] = -1
    return torch.arange(1, tensor.shape[dim] + 1, device=tensor.device, dtype=tensor.dtype).view(view)


def softmax_beta(inputs: torch.Tensor, dim: int = -1, beta: float = 1.0) -> torch.Tensor:
    resolved_beta = max(float(beta), 1e-6)
    return torch.softmax(inputs * resolved_beta, dim=dim)


def sparsemax(inputs: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    if dim < 0:
        dim = inputs.dim() + dim

    sorted_inputs, _ = torch.sort(inputs, dim=dim, descending=True)
    rho = _make_ix_like(sorted_inputs, dim)
    support = 1 + rho * sorted_inputs > sorted_inputs.cumsum(dim=dim)
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (sorted_inputs.cumsum(dim=dim).gather(dim, support_size - 1) - 1.0) / support_size
    output = torch.clamp(inputs - tau, min=0.0)
    normalizer = output.sum(dim=dim, keepdim=True).clamp(min=eps)
    return output / normalizer


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


def entmax_bisect(
    inputs: torch.Tensor,
    alpha: float = 1.5,
    dim: int = -1,
    n_iter: int = 50,
    eps: float = 1e-12,
) -> torch.Tensor:
    resolved_alpha = float(alpha)
    if resolved_alpha <= 1.0 + 1e-6:
        return softmax_beta(inputs, dim=dim, beta=1.0)
    if abs(resolved_alpha - 1.5) < 1e-6:
        return entmax15(inputs, dim=dim, eps=eps)
    if abs(resolved_alpha - 2.0) < 1e-6:
        return sparsemax(inputs, dim=dim, eps=eps)

    if dim < 0:
        dim = inputs.dim() + dim

    power = 1.0 / (resolved_alpha - 1.0)
    scaled = (resolved_alpha - 1.0) * inputs
    max_values = scaled.max(dim=dim, keepdim=True).values
    tau_lo = max_values - 1.0
    tau_hi = max_values

    for _ in range(max(int(n_iter), 1)):
        tau_mid = (tau_lo + tau_hi) / 2.0
        probabilities = torch.clamp(scaled - tau_mid, min=0.0).pow(power)
        sums = probabilities.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(sums >= 1.0, tau_mid, tau_lo)
        tau_hi = torch.where(sums < 1.0, tau_mid, tau_hi)

    output = torch.clamp(scaled - (tau_lo + tau_hi) / 2.0, min=0.0).pow(power)
    normalizer = output.sum(dim=dim, keepdim=True).clamp(min=eps)
    return output / normalizer


def bounded_confidence_activation(
    inputs: torch.Tensor,
    dim: int = -1,
    gamma: float = 8.0,
    tau: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    shifted = inputs - float(tau)
    resolved_gamma = max(float(gamma), 1e-6)
    gated = torch.sigmoid(resolved_gamma * shifted) * F.softplus(shifted)
    normalizer = gated.sum(dim=dim, keepdim=True).clamp(min=eps)
    return gated / normalizer


def apply_simplex_activation(
    inputs: torch.Tensor,
    family: str = "entmax",
    dim: int = -1,
    alpha: float = 1.5,
    beta: float = 1.0,
    tau: float = 0.0,
    gamma: float = 8.0,
    n_iter: int = 50,
    eps: float = 1e-12,
) -> torch.Tensor:
    resolved_family = family.lower()
    if resolved_family == "softmax":
        return softmax_beta(inputs, dim=dim, beta=beta)
    if resolved_family == "sparsemax":
        return sparsemax(inputs, dim=dim, eps=eps)
    if resolved_family in {"bounded_confidence", "smooth_bounded_confidence"}:
        return bounded_confidence_activation(inputs, dim=dim, gamma=gamma, tau=tau, eps=eps)
    return entmax_bisect(inputs, alpha=alpha, dim=dim, n_iter=n_iter, eps=eps)


class Entmax15(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return entmax15(inputs, dim=self.dim)
