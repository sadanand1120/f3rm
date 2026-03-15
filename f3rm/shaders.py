import torch
from torch import nn, Tensor


class ProbFromProbsShader(nn.Module):
    """Visualize probabilities (already in [0,1]) as grayscale without extra sigmoid."""

    def __init__(self):
        super().__init__()

    def forward(self, probs: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        p = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        p = torch.clamp(p, 0.0, 1.0)
        rgb = p.expand(*p.shape[:-1], 3)
        if valid_mask is not None:
            rgb = rgb * valid_mask.to(rgb.device)
        return rgb.to(torch.float16)
