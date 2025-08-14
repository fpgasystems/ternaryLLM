import torch
import time

# Set device to CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the size of the sparse tensor and density
rows, cols = 8192, 2048  # Large dimensions
sparsity = 0.3          # Proportion of non-zero elements

# Number of non-zero elements
num_nonzero = int(rows * cols * sparsity)

# Generate random indices for non-zero elements
indices = torch.randint(0, rows, (2, num_nonzero), device='cpu')  # shape: (2, num_nonzero)

# Generate random values for the non-zero elements
values = torch.randn(num_nonzero, device='cpu')

# Create the sparse tensor and move it to CUDA
sparse_tensor = torch.sparse_coo_tensor(indices, values, (rows, cols)).to(device)

# Generate a large dense matrix on CUDA
dense_matrix = torch.randn(cols, 256, device=device)  # Output will have shape (rows, 512)


start = time.time()
for i in range(5):

    start_event = torch.cuda.Event(enable_timing=True, blocking=True)
    end_event = torch.cuda.Event(enable_timing=True, blocking=True)

    # Perform sparse-dense matrix multiplication
    start_event.record()
    result = torch.sparse.mm(sparse_tensor, dense_matrix)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    print(elapsed_time)
    # Print result shape to verify
    print("Result shape:", result.shape)
end = time.time()
print(f"end-end: {(end-start)*1000}")