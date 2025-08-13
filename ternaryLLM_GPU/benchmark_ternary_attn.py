import torch, json, gc
import numpy as np
from TernaryLLM import TernaryConfig, LlamaTernaryAttention
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding


def benchmark_attn(config: TernaryConfig, ternarization: bool = True):
    if ternarization:
        attn_layer = LlamaTernaryAttention(config, 0).to('cuda')
    else:
        attn_layer = LlamaAttention(config, 0).to('cuda')
        # attn_layer = LlamaSdpaAttention(llama_config, 0).to('cuda')
    
    torch.cuda.init()
    torch.cuda.synchronize()
    

    seq_len = 256
    input_hidden_states = torch.rand((1, seq_len, llama_config.hidden_size)).to('cuda')
    rotary_emb = LlamaRotaryEmbedding(llama_config).to('cuda')
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).to('cuda')
    cos, sin = rotary_emb(input_hidden_states, position_ids)
    position_embeddings = (cos, sin)

    attn_spans = []
    attn_mems = []

    # warm-up
    for i in range(10):
        with torch.no_grad():
            output = attn_layer(hidden_states=input_hidden_states, position_embeddings=position_embeddings, attention_mask=None)
    # torch.cuda.empty_cache()


    for i in range(20):
        start_event = torch.cuda.Event(enable_timing=True, blocking=True)
        end_event = torch.cuda.Event(enable_timing=True, blocking=True)
        start_event.record()

        with torch.no_grad():
            output = attn_layer(hidden_states=input_hidden_states.detach(), position_embeddings=position_embeddings,  attention_mask=None)
        reserv_mem = torch.cuda.memory_reserved(0)
        alloc_mem = torch.cuda.memory_allocated(0)

        end_event.record()
        torch.cuda.synchronize()

        # collect stat
        elapsed_time = start_event.elapsed_time(end_event)
        attn_spans.append(elapsed_time)
        attn_mems.append(alloc_mem)
        
        del output
    torch.cuda.empty_cache()
    
    attn_spans.pop(0)
    attn_spans = np.array(sorted(attn_spans, reverse=(not ternarization)))
    
    return attn_spans, np.array(attn_mems)
    

if __name__ == "__main__":
    with open('./config/config.json') as f:
        llama_3_1b_json = json.load(f)
    
    llama_config = TernaryConfig(
        vocab_size=llama_3_1b_json["vocab_size"],
        hidden_size=llama_3_1b_json["hidden_size"],
        intermediate_size=llama_3_1b_json["intermediate_size"],
        attention_dropout=llama_3_1b_json["attention_dropout"],
        num_attention_heads=llama_3_1b_json["num_attention_heads"],
        head_dim=llama_3_1b_json["head_dim"],
        num_key_value_heads=llama_3_1b_json["num_key_value_heads"],
        max_position_embeddings=llama_3_1b_json["max_position_embeddings"],
        rope_theta=llama_3_1b_json["rope_theta"],
        rope_scaling=llama_3_1b_json["rope_scaling"],
        sparsity=llama_3_1b_json["sparsity"],
        uniform_sparsity=llama_3_1b_json["uniform_sparsity"],
        uniform_sparsity_block_size=llama_3_1b_json["uniform_sparsity_block_size"],
        padding=llama_3_1b_json["padding"],
        padding_size=llama_3_1b_json["padding_size"]
    )

    sparsities = [i/100 for i in range(70, 100, 1)]
    sparsities.append(0.999)

    import sys
    ternary_benchmark = bool(int(sys.argv[1])) if len(sys.argv) >= 2 else False
    log = bool(int(sys.argv[2])) if len(sys.argv) >= 3 else False
    sparsities = [float(sys.argv[3])] if len(sys.argv) >= 4 else sparsities


    seq_len = 256
    log_ter_time = open(f'./data/llama_3_1b_ter_attn_nonuniform_time_{seq_len}', "w")
    log_ter_mem = open(f'./data/llama_3_1b_ter_attn_nonuniform_mem_{seq_len}', "w")
    log_vanila_time = open(f'./data/llama_3_1b_vanila_attn_time_{seq_len}', "w")
    log_vanila_mem = open(f'./data/llama_3_1b_vanila_attn_mem_{seq_len}', "w")

    for sp in sparsities:
        llama_config.sparsity = sp
        ter_attn_spans, ter_attn_mems = benchmark_attn(llama_config, True)
        print(f"TerSpMM {sp} {ter_attn_spans[0:10].mean()} {ter_attn_mems.mean()/1024/1024}MB")
        log_ter_time.write(f"{sp} {ter_attn_spans[0:10].mean()}\n")
        log_ter_mem.write(f"{sp} {ter_attn_mems.mean()}\n")
        torch.cuda.empty_cache()

        vanila_attn_spans, vanila_attn_mems = benchmark_attn(llama_config, False)
        print(f"nnLinear {sp} {vanila_attn_spans[0:10].mean()} {vanila_attn_mems.mean()/1024/1024}MB")
        log_vanila_time.write(f"{sp} {vanila_attn_spans[0:10].mean()}\n")
        log_vanila_mem.write(f"{sp} {vanila_attn_mems.mean()}\n")

        torch.cuda.empty_cache()
        gc.collect()
    
    log_ter_time.close()
    log_ter_mem.close()
    log_vanila_time.close()
    log_vanila_mem.close()