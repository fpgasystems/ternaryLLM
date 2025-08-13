import torch
import torch.utils.checkpoint
from torch import nn


from TernaryLLM.configuration_ternary import TernaryConfig
from transformers.activations import ACT2FN
from .TernaryLinear import TernaryLinear


class LlamaTernaryMLP(nn.Module):
    """Refer to LlamaMLP
    """
    def __init__(self, config: TernaryConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = TernaryLinear(self.hidden_size, self.intermediate_size, 
                                       sparsity=config.sparsity, uniform=config.uniform_sparsity, uniform_blk_size=config.uniform_sparsity_block_size, 
                                       padding=config.padding, padding_size=config.padding_size,
                                       bias=config.mlp_bias)
        self.up_proj = TernaryLinear(self.hidden_size, self.intermediate_size, 
                                     sparsity=config.sparsity, uniform=config.uniform_sparsity, uniform_blk_size=config.uniform_sparsity_block_size, 
                                     padding=config.padding, padding_size=config.padding_size,
                                     bias=config.mlp_bias)
        self.down_proj = TernaryLinear(self.intermediate_size, self.hidden_size, 
                                       sparsity=config.sparsity, uniform=config.uniform_sparsity, uniform_blk_size=config.uniform_sparsity_block_size, 
                                       padding=config.padding, padding_size=config.padding_size,
                                       bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

__all__ = ["LlamaTernaryMLP"]