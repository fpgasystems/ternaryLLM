from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import nn
from .TernaryLinear import TernaryLinear
from .configuration_ternary import TernaryConfig

def pad_sequence():
    pass

def collect_linear_layers(model, config: TernaryConfig):
    layers_to_replace = []
    for name, module in model.named_modules():
        if config.ternary_attn_linear and isinstance(module, nn.Linear) and ("q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name):
            layers_to_replace.append((name, module))
        
        if config.ternary_mlp and isinstance(module, nn.Linear) and ("mlp" in name):
            layers_to_replace.append((name, module))
    return layers_to_replace
        

def replace_linear_layers(layers, model, config: TernaryConfig):
    for name, module in layers: 
        print(f"Current Layer Name: {name}, Module: {module}")

        parent_module_name, core_module_name = name.rsplit('.', 1)    # parent module is the common part, core module is the module to replace
        print(f"Parent Module Name: {parent_module_name}, Core Module Name: {core_module_name}")

        ternary_layer = \
            TernaryLinear(
                module.in_features, module.out_features, 
                sparsity=config.sparsity, 
                uniform=config.uniform_sparsity, uniform_blk_size=config.uniform_sparsity_block_size, 
                padding=False, bias=False, device='cpu')  # put on cpu first

        parent_module = model.get_submodule(parent_module_name)
        setattr(parent_module, core_module_name, ternary_layer)


def prepare_padded_embeds_input(model, tokenizer, seq_len):
    input_ids = tokenizer(["hello"] * seq_len, add_special_tokens=False)["input_ids"]
    custom_inputs = tokenizer.pad(
        {"input_ids": [input_ids]}, 
        return_tensors="pt", 
        padding="max_length", 
        max_length=seq_len
    ).to('cuda')

    # Get input embeddings from the model
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(custom_inputs["input_ids"]).squeeze(2)

    return embeddings

def add_padding_to_token(input_ids, tokenizer, alignment_size: int = 4, device='cuda'):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    seq_length = input_ids.size(1)         # Get sequence length
    remainder = seq_length % alignment_size
    if remainder != 0:
        padding_needed = alignment_size - remainder
        pad_token_id = tokenizer.pad_token_id  # Get the pad token ID
        pad_tensor = torch.full((input_ids.size(0), padding_needed), pad_token_id, dtype=torch.long)
        input_ids = torch.cat((input_ids, pad_tensor), dim=1)  # Concatenate padding
    input_ids = input_ids.to(device)
    torch.cuda.empty_cache()
    return input_ids


def prepare_ternary_model(model, config: TernaryConfig, device='cuda'):
    model.to('cpu') # move to cpu for ternarization
    attn_linear_layers = collect_linear_layers(model, config)
    replace_linear_layers(attn_linear_layers, model, config)
    model.to(device)
    torch.cuda.empty_cache()
    return model