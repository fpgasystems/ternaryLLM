import torch
from torch import nn, tensor
import time
import numpy as np
from torcheval.metrics import Throughput

class ToyModel(nn.Module):
    def __init__(self, hidden_size: int = 10, output_dim: int = 5):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(hidden_size, hidden_size)
        self.net2 = nn.Linear(hidden_size, hidden_size)
        self.net3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        return self.net3(self.net2(self.net1(x)))


if __name__ == "__main__":
    batch = 1
    rows = 256
    columns = 8192
    inners = 2048
    data1 = torch.rand(batch, rows, columns).to('cuda')
    net1 = nn.Linear(columns, inners, True).to('cuda')
    torch.cuda.init()
    torch.cuda.synchronize()

    total_kn_span = 0
    kn_spans = []


    # metric = Throughput()
    # items_processed = 128
    # ts = time.monotonic()

    print(data1.size())
    for i in range(50):
        start_w_overhead = time.time()
        # torch.cuda.reset_peak_memory_stats()
        # torch.cuda.empty_cache()
        
        torch.cuda.synchronize()


        start_event = torch.cuda.Event(enable_timing=True, blocking=True)
        end_event = torch.cuda.Event(enable_timing=True, blocking=True)

        start_event.record()

        output = net1(data1)
        # output = model(data1)
        # output = torch.mm(data1, data2)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)
        end_w_overhead = time.time()
        # print(f"end-end: {(end_w_overhead-start_w_overhead)*1000}")
        print(elapsed_time)
        # print(torch.cuda.memory_summary())
        total_kn_span += elapsed_time if i > 0 else 0
        kn_spans.append(elapsed_time)
    print(f"Avg: {total_kn_span/49}")
    kn_spans.pop(0)
    
    kn_spans = np.array(sorted(kn_spans, reverse=True))
    print(kn_spans[0:10].mean())

    # elapsed_time_sec = time.monotonic() - ts
    # metric.update(items_processed, elapsed_time_sec)
    # print(metric.compute())