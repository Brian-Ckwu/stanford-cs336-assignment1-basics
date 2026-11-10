import torch
import torch.nn as nn
from jaxtyping import Float

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        raise NotImplementedError

    def forward(
        self,
        x: Float[torch.Tensor, "*batch in_features"],
    ) -> Float[torch.Tensor, "*batch out_features"]:
        raise NotImplementedError
