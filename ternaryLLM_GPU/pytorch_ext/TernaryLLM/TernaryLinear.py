import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
import numpy as np
import time
import ter_spmm


MapKSizeToTileSize = {
    8192: 512,
    4096: 512,
    3072: 512,
    2048: 256,
    1024: 256,
    512: 64
}


def ternarize_weights(w: torch.Tensor):
    raise NotImplementedError("ternarize_weights not implement")


def random_ternary_weights(
    size: tuple, 
    sparsity: float = 0.8, 
    uniform: bool = False, 
    block_size: int = 512):
    """generate random ternary weight

    Args:
        size (tuple): (in_features/inners/K, out_features/columns/N)
        sparsity (float, optional): Defaults to 0.8.
        uniform (bool, optional): If True, ensures that 1 and -1 are perfectly balanced and uniformly distributed. Defaults to False.

    Returns:
        tensor: ternary weight tensor
    """
    w = None
    if uniform:
        total_elements = size[0] * size[1]
        num_blocks = total_elements // block_size
        total_nnz = int(total_elements * (1 - sparsity))

        # ensure total_nnz is even for equal split of +1/-1
        if total_nnz % 2 != 0:
            total_nnz -= 1

        # initialize nnz count per block with zeros
        base_nnz = (total_nnz // num_blocks) // 2 * 2  # make it even
        nnz_counts = torch.full((num_blocks,), base_nnz, dtype=torch.int32)
        
        # distribute any remaining nnz elements (must be even and balanced)
        remaining = total_nnz - base_nnz * num_blocks
        for i in range(0, remaining, 2):
            nnz_counts[i % num_blocks] += 2

        flat_tensor = torch.zeros(total_elements, dtype=torch.int32)

        for i in range(num_blocks):
            nnz = nnz_counts[i].item()
            if nnz == 0:
                continue

            half_nnz = nnz // 2
            start = i * block_size

            # select nnz unique positions in the block
            indices = torch.randperm(block_size)[:nnz]
            block_indices = start + indices

            # generate equal number of +1 and -1
            signs = torch.cat([torch.ones(half_nnz), -torch.ones(half_nnz)])
            signs = signs[torch.randperm(nnz)]

            flat_tensor[block_indices] = signs.int()
        
        w = flat_tensor.view((size[1], size[0])).T.contiguous()
    else:
        rand_tensor = torch.rand(size)
        w = torch.zeros(size, dtype=torch.int32, requires_grad=False)
        w[rand_tensor < sparsity] = 0       
        w[(rand_tensor >= sparsity) & (rand_tensor < (sparsity+1)/2)] = -1  
        w[rand_tensor >= (sparsity+1)/2] = 1  
        del rand_tensor
    return w

def convert_ternary_weights(
    w: torch.Tensor, 
    padding: bool = False,
    padding_size: int = 4,
    dtype = torch.int32,
    device = 'cpu'
):
    """Convert a ternary weight to Merged CSC Format

    Args:
        w (torch.Tensor): (inners/K, columns/N)
        device (str, optional): Defaults to 'cpu' because only cpu side algorithm is implemented.

    Returns:
        row_indices, col_offsets, col_tile_indices
    """
    w_ = w.detach()
    cnt_pos = (w_ == 1).sum().item()
    cnt_neg = (w_ == -1).sum().item()
    k = w_.size()[0]
    tmcsc = ter_spmm.convert_to_ter_tiled_mcsc(w_, w_.size(1), w_.size(0), cnt_pos, cnt_neg, MapKSizeToTileSize[k], padding, padding_size)
    del w, w_
    
    row_indices = torch.tensor(np.array(tmcsc[0]), dtype=torch.int32, requires_grad=False, device=device).contiguous().to(dtype)
    col_offsets = torch.tensor(np.array(tmcsc[1]), dtype=torch.int32, requires_grad=False, device=device).contiguous()
    col_tile_indices = torch.tensor(np.array(tmcsc[2]), dtype=torch.int32, requires_grad=False, device=device).contiguous().to(dtype)
    return row_indices, col_offsets, col_tile_indices

class TernaryLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    w_row_indices: Tensor
    w_col_offsets: Tensor
    w_col_tile_indices: Tensor
    def __init__(
        self, 
        in_features: int,   # inners, K
        out_features: int,  # columns, N
        sparsity: float = 0.8,
        uniform: bool = False,
        uniform_blk_size: int = 512,
        padding: bool = False,
        padding_size: int = 4,
        bias: bool = True,
        device=None,
        dtype=torch.int32,) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.ternary_weight = random_ternary_weights((in_features, out_features), sparsity, uniform, uniform_blk_size)
        w_row_indices, w_col_offsets, w_col_tile_indices = convert_ternary_weights(self.ternary_weight, padding=padding, padding_size=padding_size, device=device, dtype=torch.int16)
        self.register_buffer("w_row_indices", w_row_indices, False)
        self.register_buffer("w_col_offsets", w_col_offsets, False)
        self.register_buffer("w_col_tile_indices", w_col_tile_indices, False)
        self.in_features = in_features
        self.out_features = out_features
        self.uniform = uniform
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        """forward of ternary linear

        Args:
            input (Tensor): (batch size, rows(m), inners(k)) | (batch size, inners(k), rows(m))

        Returns:
            Tensor: results (batch size, rows(m), columns(n))
        """
        assert len(input.shape) <= 3, f"Input has dim larger than 3"
        
        # if input.shape[-1] == self.in_features:
        #     assert input.stride()[1] == 1, "The input of shape (bs, m, k) should be stored in column major"
        # if input.shape[-2] == self.in_features:
        #     assert input.is_contiguous(), "The input of shape (bs, k, m) should be contiguous"
        
        batch_size = input.shape[0] if len(input.shape) == 3 else 0
        rows = input.shape[-2] if input.shape[-1] == self.in_features else input.shape[-1]
        res = ter_spmm.ter_spmm(
            input,
            self.w_row_indices,
            self.w_col_offsets,
            self.w_col_tile_indices,
            batch_size,         # batch size
            rows,               # rows, M
            self.out_features,  # columns, N
            self.in_features,   # inners, K
            self.uniform,       # perfectly uniformed and balanced +1/-1 distribution
            self.padding        # padded
        )
        return res

__all__ = ["TernaryLinear"]