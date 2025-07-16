from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from .configuration_ternary import TernaryConfig
from .TernaryLinear import TernaryLinear
import math


class LlamaTernaryAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: TernaryConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            print(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = TernaryLinear(self.hidden_size, self.num_heads * self.head_dim, 
                                    sparsity=config.sparsity, uniform=config.uniform_sparsity, uniform_blk_size=config.uniform_sparsity_block_size, 
                                    padding=config.padding, padding_size=config.padding_size,
                                    bias=config.attention_bias)
        self.k_proj = TernaryLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, 
                                    sparsity=config.sparsity, uniform=config.uniform_sparsity, uniform_blk_size=config.uniform_sparsity_block_size, 
                                    padding=config.padding, padding_size=config.padding_size,
                                    bias=config.attention_bias)
        self.v_proj = TernaryLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, 
                                    sparsity=config.sparsity, uniform=config.uniform_sparsity, uniform_blk_size=config.uniform_sparsity_block_size, 
                                    padding=config.padding, padding_size=config.padding_size,
                                    bias=config.attention_bias)
        self.o_proj = TernaryLinear(self.num_heads * self.head_dim, self.hidden_size, 
                                    sparsity=config.sparsity, uniform=config.uniform_sparsity, uniform_blk_size=config.uniform_sparsity_block_size, 
                                    padding=config.padding, padding_size=config.padding_size,
                                    bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config, device='cuda')

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            # logger.warning_once(
            #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            #     "removed and `position_embeddings` will be mandatory."
            # )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


"""
    Only used following function to replace nn.Linear in self attention layer within LlaMa Model
"""

def collect_attn_linear_layers(model):
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name):
            layers_to_replace.append((name, module))
    
    return layers_to_replace

def replace_linear_layers(layers, attn, sparsity = 0.8):
    for name, module in layers: 
        # print(f"Current Layer Name: {name}, Module: {module}")
        ternary_layer = TernaryLinear(module.in_features, module.out_features, sparsity, False).to('cuda')
        setattr(attn, name, ternary_layer)
    torch.cuda.empty_cache()

__all__ = ["LlamaTernaryAttention"]