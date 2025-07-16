from transformers.models.llama.modeling_llama import LlamaMLP
from TernaryLLM import LlamaTernaryMLP, TernaryConfig

import torch, json, time, sys
import numpy as np

def benchmark_mlp(config: TernaryConfig, ternarization: bool = True):
    mlp_layer = None
    if ternarization:
        mlp_layer = LlamaTernaryMLP(config).to('cuda')
    else:
        mlp_layer = LlamaMLP(config).to('cuda')

    torch.cuda.init()
    torch.cuda.synchronize()

    seq_len = 256
    input = torch.randn((1, seq_len, config.hidden_size)).to('cuda')

    mlp_cpu_spans = []
    mlp_spans = []
    mlp_mems = []

    # warm-up
    for _ in range(10):
        with torch.no_grad():
            output = mlp_layer(input)
    
    # benchmark
    for i in range(100):
        start_event = torch.cuda.Event(enable_timing=True, blocking=True)
        end_event = torch.cuda.Event(enable_timing=True, blocking=True)
        start_event.record()
        start_cpu = time.time()

        with torch.no_grad():
            mlp_layer(input)
        alloc_mem = torch.cuda.memory_allocated(0)

        end_event.record()
        torch.cuda.synchronize()
        end_cpu = time.time()

        # collect stat
        elapsed_time = start_event.elapsed_time(end_event)
        elapsed_cpu_time = end_cpu - start_cpu
        mlp_spans.append(elapsed_time)
        mlp_mems.append(alloc_mem)
        mlp_cpu_spans.append(elapsed_cpu_time*1000)
    
    mlp_spans.pop(0)
    mlp_spans = np.array(sorted(mlp_spans, reverse=(not ternarization)))
    mlp_cpu_spans = np.array(sorted(mlp_cpu_spans, reverse=(not ternarization)))

    return mlp_cpu_spans, mlp_spans, np.array(mlp_mems)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python3 benchmark <Ternary: 1/0> <Sparsity: Optional>")

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
        hidden_act=llama_3_1b_json["hidden_act"],
        mlp_bias=llama_3_1b_json["mlp_bias"],
        sparsity=llama_3_1b_json["sparsity"],
        uniform_sparsity=llama_3_1b_json["uniform_sparsity"],
        uniform_sparsity_block_size=llama_3_1b_json["uniform_sparsity_block_size"],
        padding=llama_3_1b_json["padding"],
        padding_size=llama_3_1b_json["padding_size"]
    )

    ternary_benchmark = bool(int(sys.argv[1])) if len(sys.argv) >= 2 else False
    log = bool(int(sys.argv[2])) if len(sys.argv) >= 3 else False
    sparsities = [i/100 for i in range(70, 100, 1)]
    sparsities.append(0.999)
    sparsities = [float(sys.argv[3])] if len(sys.argv) >= 4 else sparsities

    if log:
        log_time = open(f'./data/llama_3_1b_ter_mlp_uniform_time_{256}', "w") if ternary_benchmark else open('./data/llama_3_1b_vanila_mlp_time', "w")
        log_mem = open(f'./data/llama_3_1b_ter_mlp_uniform_mem_{256}', "w") if ternary_benchmark else open('./data/llama_3_1b_vanila_mlp_mem', "w")

    for sp in sparsities:
        llama_config.sparsity = sp
        mlp_cpu_spans, mlp_spans, mlp_mems = benchmark_mlp(llama_config, ternary_benchmark)
        print(f"{sp} {mlp_spans[0:5].mean()} {mlp_mems.mean()/1024/1024}MB")

        if log:
            log_time.write(f"{sp} {mlp_spans[0:10].mean()}\n")
            log_mem.write(f"{sp} {mlp_mems.mean()}\n")

        torch.cuda.empty_cache()

    if log:
        log_time.close()
        log_mem.close()