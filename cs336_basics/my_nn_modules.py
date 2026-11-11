import torch
import einops
import torch.nn as nn
from jaxtyping import Float

class Linear(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        data = torch.empty(size=(d_out, d_in), dtype=dtype, device=device)  # create the data directly on device to avoid data moving overhead
        std = (2 / (d_in + d_out))**0.5
        nn.init.trunc_normal_(tensor=data, mean=0, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(data=data)  # .to(device) return a "Tensor", so do not call it here

    def forward(
        self,
        x: Float[torch.Tensor, "*batch d_in"],
    ) -> Float[torch.Tensor, "*batch d_out"]:
        return einops.einsum(
            x, self.weight, "... d_in, d_out d_in -> ... d_out"
        )  # Need to use self.weight since self.weight.data bypasses the autograd engine
