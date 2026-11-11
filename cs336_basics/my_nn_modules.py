import torch
import einops
import torch.nn as nn
from jaxtyping import Float, Int

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

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,  # |V|
        embedding_dim: int,  # d_model
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        data = torch.empty(size=(num_embeddings, embedding_dim), dtype=dtype, device=device)  # create the data directly on device to avoid data moving overhead
        nn.init.trunc_normal_(tensor=data, mean=0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(data=data)  # .to(device) return a "Tensor", so do not call it here

    def forward(
        self,
        token_ids: Int[torch.Tensor, "batch_size sequence_length"]
    ) -> Float[torch.Tensor, "batch_size sequence_length embedding_dim"]:
        # Paralleled selection of embedding vectors for all batch_size * sequence_length operations
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            data=torch.ones(size=(d_model,), device=device, dtype=dtype)
        )

    def forward(
        self,
        x: Float[torch.Tensor, "B L d"]  # batch_size, sequence_length, d_model
    ) -> Float[torch.Tensor, "B L d"]:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"The last dimension of x (shape={x.shape}) should be {self.d_model}")
        in_dtype = x.dtype
        x = x.to(torch.float32)  # float16 can only represent +-65504, so it can easily overflow when an activation exceeds 256
        rms = ((x**2).sum(dim=-1) / self.d_model + self.eps)**0.5  # Since RMS should be calculated from each token vector, `rms` should have shape (B, L) if x has shape (B, L, d)
        rms = einops.rearrange(rms, "... -> ... 1")  # rearrange for broadcasting
        out = x / rms * self.weight
        return out.to(in_dtype)  # remember to cast back to the original dtype
