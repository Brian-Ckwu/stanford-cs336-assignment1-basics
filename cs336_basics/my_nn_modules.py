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
