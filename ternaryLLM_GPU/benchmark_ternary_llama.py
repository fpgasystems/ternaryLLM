import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from TernaryLLM.TernaryLlama import (
    add_padding_to_token,
    prepare_ternary_model
)
from TernaryLLM.configuration_ternary import TernaryConfig
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, time, sys, json
import numpy as np
from accelerate import init_empty_weights
import subprocess, threading


power_monitor_lock = threading.Lock()
power_monitor = False
total_energy_joules = 0
walt_trends = []
def collect_power(interval: float = 0.1):
    global total_energy_joules
    global power_monitor
    global power_monitor_lock
    global walt_trends
    while True:
        with power_monitor_lock:
            if not power_monitor:
                break
        time.sleep(interval)
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,nounits"],
            capture_output=True,
            text=True,
        )
        power_w = float(output.stdout.split("\n")[1])
        walt_trends.append(power_w)
        total_energy_joules += power_w * interval

def benchmark_llama_raw(
    model, 
    tokenizer, 
    config: TernaryConfig, 
    text: str, 
    interval: float = 0.1,
    ternarization: bool = True
):
    model = prepare_ternary_model(model, config) if ternarization else model
    model.to('cuda')    # move to gpu

    # Tokenize the input string and add paddings to tokens
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = add_padding_to_token(input_ids, tokenizer, 4, 'cuda')
    print(f"Sequence length: {input_ids.size(1)}")

    with torch.no_grad():
        for _ in range(5):
            output = model(input_ids)
    torch.cuda.empty_cache()

    # prepare power benchmark
    global total_energy_joules
    global power_monitor
    global power_monitor_lock
    global walt_trends
    # start power measurement
    with power_monitor_lock:
        power_monitor = True
    power_monitor_thread = threading.Thread(target=collect_power, args=(interval,), daemon=True)
    power_monitor_thread.start()

    # prepare time benchmark
    gen_mems = []
    total_tokens = 0
    cpu_ts = time.monotonic()

    # start generation
    with torch.no_grad():
        for i in range(1):

            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=500,
                do_sample=False,
                use_cache=False,
            )
            alloc_mem = torch.cuda.memory_allocated(0)

            # Decode the output tokens to a string
            print(output.shape[-1])
            num_tokens = output.shape[-1] - input_ids.size(1)
            total_tokens += num_tokens
            gen_mems.append(alloc_mem)
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed_time_sec = time.monotonic() - cpu_ts
    gen_mems = np.array(gen_mems)
    
    # end power measurement
    with power_monitor_lock:
        power_monitor = False
    power_monitor_thread.join()
    walt_trends = np.array(walt_trends)

    return (total_tokens, total_tokens/elapsed_time_sec, (25*input_ids.size(1))/elapsed_time_sec, gen_mems, walt_trends, total_energy_joules)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python3 benchmark <Ternary: 1/0>")
    ternary_benchmark = bool(int(sys.argv[1])) if len(sys.argv) >= 2 else False

    # load ternary model config
    with open('./config/config.json') as f:
        llama_3_1b_json = json.load(f)
    ternary_llama_config = TernaryConfig(
        ternary_attn_linear=llama_3_1b_json["ternary_attn_linear"],
        ternary_mlp=llama_3_1b_json["ternary_mlp"],
        sparsity=llama_3_1b_json["sparsity"],
        uniform_sparsity=llama_3_1b_json["uniform_sparsity"],
        uniform_sparsity_block_size=llama_3_1b_json["uniform_sparsity_block_size"],
        padding=llama_3_1b_json["padding"],
        padding_size=llama_3_1b_json["padding_size"]
    )

    # load and prepare float model
    model_name = "meta-llama/Llama-3.2-1B"
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
    model.tie_weights()
    model.to_empty(device="cpu")  # Allocates empty tensors on GPU
    model.load_state_dict(model.state_dict())  # Re-initializes weights
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load the dataset from ag news
    dataset = load_dataset("fancyzhx/ag_news")

    # prepare log file
    log_time = open('./data/llama_3_1b_ter_llama_generation', "a") if ternary_benchmark else open('./data/llama_3_1b_vanilla_llama_generation', "w")

    total_tokens, output_token_throughput, input_token_throughput, mems, walts, total_engry = \
        benchmark_llama_raw(
            model, tokenizer, ternary_llama_config, 
            text=dataset["train"][0]['text'], 
            interval=0.1, ternarization=ternary_benchmark)
    print(f"Total Generated Token: {total_tokens} Throughput: {output_token_throughput} {input_token_throughput} | Memory: {mems.mean()/1024/1024}MB")
    print(f"Power consumption: {total_engry} joules; Avg Watt: {walts.mean():.2f}W")
    log_time.write(f"{llama_3_1b_json["sparsity"]} {output_token_throughput}\n")
    torch.cuda.empty_cache()
    
    log_time.close()