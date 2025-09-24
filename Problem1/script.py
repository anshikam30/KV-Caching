import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Given Specifications
# -------------------------------------------------------
L = 32                       # Number of layers
QH = 32                      # Number of Query Heads
KVH = 8                      # Number of KV Heads
h1 = 4096                    # Embedding dimension
h2 = 14336                   # Inner dimension (not used in these calcs)
V = 128256                   # Vocab size
GPU_memory_GB = 80           # GPU memory in GB
S = 4096                     # Context Length
WB = 2                       # Weight Bytes (Bytes per parameter) assumed to be FP16
KVB = 2                      # KV cache Bytes (Bytes per key/value element)
Param_count = 8_000_000_000  # Total parameters in model


# -------------------------------------------------------
# Problem 1.1 - Parameter Memory and KV Cache Calculation
# -------------------------------------------------------

# Total Parameter Memory in GB
total_paramers_mem_bytes = Param_count * WB
total_parameters_mem_GB = total_paramers_mem_bytes / (1024 ** 3)  #Divide by 1024^3 to get the corresponding GBs

# KV Cache per request (in GBytes)
num_attn_heads = h1/QH

kv_cache_bytes = 2*S * L * KVH * KVB * num_attn_heads       #for each K and V, layer and token: 
kv_cache_GB = kv_cache_bytes / (1024 ** 3)                   #Divide by 1024^3 to get the corresponding GBs

# Maximum permissible batch size is given by mem available for Storing KV cache/1 KV cache size
max_batch_size = (GPU_memory_GB - total_parameters_mem_GB) / kv_cache_GB

# Print results
print(f"Total Parameters : {Param_count:} ")
print(f"Total Parameter Memory (GB): {total_parameters_mem_GB:.2f} GB")
print(f"KV Cache per request (GB): {kv_cache_GB:.2f} GB")
print(f"Max Batch Size: {int(max_batch_size)}")

# --------------------------------------------------------
# Problem 1.2 - Tensor Parallelism
# --------------------------------------------------------

def tensor_parallelism(X):
    # Model parameters and KV cache shrink by factor of X
    param_mem_GB_X = total_parameters_mem_GB / X
    kv_cache_GB_X = kv_cache_GB / X                   #Divide by 1024^3 to get the corresponding GBs
    # Maximum permissible batch size per GPU is given by mem available for Storing KV cache/1 KV cache size
    max_batch_size_X = (GPU_memory_GB - param_mem_GB_X) / kv_cache_GB_X
    total_batch_size = max_batch_size_X*X
    return total_batch_size,param_mem_GB_X, kv_cache_GB_X, max_batch_size_X

# Example for X = 3
total_batch_size, param_mem_GB_X, kv_cache_GB_X, max_batch_size_X = tensor_parallelism(3)
print(f"\nWith Tensor Parallelism of dimension X = 3:")
print(f"Total Parameters : {Param_count:} ")
print(f" - Parameter Memory per GPU: {param_mem_GB_X:.2f} GB")
print(f" - KV Cache per request per GPU: {kv_cache_GB_X:.2f} GB")
print(f" - Max Batch Size Per GPU: {int(max_batch_size_X)}")
print(f" - Max Batch Size over all GPUs: {int(total_batch_size)}")

# ----------------------------------------------------------
# Plotting Max Batch Size vs Tensor Parallelism Dimension (X)
# ----------------------------------------------------------

X_values = np.arange(1, 33, 1)  # X from 1 to 32 
max_batch_sizes = []
for X in X_values:
    _,_, _, max_batch_size_X = tensor_parallelism(X)
    max_batch_sizes.append(max_batch_size_X)
#Plot the Max Batch Size Vs Tensor Dimension Graph.
plt.figure(figsize=(8, 6))
plt.plot(X_values, max_batch_sizes, marker='o', color='green')
plt.title("Max Batch Size vs Tensor Parallelism Dimension (X)")
plt.xlabel("Tensor Parallelism Dimension (X)")
plt.ylabel("Maximum Batch Size")
plt.xticks(X_values)
plt.grid(True)
plt.savefig("MaxBatchSize_vs_TensorParallelism.png", dpi=300)
plt.show()

# ------------------------------------------------------
# Problem 1.3 & 1.4 - Arithmetic Intensity Calculations
# ------------------------------------------------------

def prefill_AI(N, d):
    # AI = N * (4d + 3) / (8 * (N + d)): Detailed explanation of how the formula was derived is given in report.
    AI = (N * (4 * d + 3)) / (8 * (N + d))
    return AI

def decode_AI(B, N, d):
    #B is the batch size, both number of flops and memory accessed scales linearly with Batch size, som AI doesn't depend on B.
    # AI = N * (4d + 3) / (4 * (Nd + d + 2N)): Detailed explanation of how the formula was derived is given in report.
    AI = (B*N * (4 * d + 3)) / (4 * ((N * d) + (B*d) + (2 *B* N)))
    return AI

# Example values N=1024, d=128
N_example = 1024
d_example = 128
prefill_ai_val = prefill_AI(N_example, d_example) #Compute AI for prefill for given N and d
decode_ai_val = decode_AI(1, N_example, d_example)   #Compute AI for decode for given N and d

print(f"\nArithmetic Intensity for Prefill Attention (N={N_example}, d={d_example}): {prefill_ai_val:.2f} FLOPs/Bytes")
print(f"Arithmetic Intensity for Decode Attention (N={N_example}, d={d_example}): {decode_ai_val:.2f} FLOPs/Bytes")

# -------------------------------------------
# Plotting AI vs N for Prefill Attention
# -------------------------------------------

N_values = [16,32,64,128,256,512,1024,2048,4196,8192] #For Various Values of N
d_fixed = 128  # fixed dimension

prefill_ai_values = [prefill_AI(N, d_fixed) for N in N_values]  #Corresponding prefill AIs for N_values
#Plot the AI Vs N graph  for prefill.
plt.figure(figsize=(8, 6))
plt.plot(N_values, prefill_ai_values, marker='o')
plt.title("Arithmetic Intensity (AI) vs Context Length (N) for Prefill Attention")
plt.xlabel("Context Length (N)")
plt.ylabel("Arithmetic Intensity (FLOPs/Bytes)")
plt.savefig("AI_vs_N_PrefillAttention.png", dpi=300)
plt.grid(True)
plt.show()

# -------------------------------------------
# Plotting AI vs N for Decode Attention
# -------------------------------------------

N_values = [16,32,64,128,256,512,1024,2048,4196,8192] #For Various Values of N
d_fixed = 128  # fixed dimension

prefill_ai_values = [decode_AI(1,N, d_fixed) for N in N_values]  #Corresponding prefill AIs for N_values
#Plot the AI Vs N graph  for prefill.
plt.figure(figsize=(8, 6))
plt.plot(N_values, prefill_ai_values, marker='o')
plt.title("Arithmetic Intensity (AI) vs Context Length (N) for Decode Attention")
plt.xlabel("Context Length (N)")
plt.ylabel("Arithmetic Intensity (FLOPs/Bytes)")
plt.savefig("AI_vs_N_DecodeAttention.png", dpi=300)
plt.grid(True)
plt.show()

# ---------------------------------------------------
# Plotting AI vs Batch Size (B) for Decode Attention
# ---------------------------------------------------
N_fixed = 1024
B_values = [2,4,8,16,32,64,128,256,512,1024,2048]  #For Various Values of B
decode_ai_values = [decode_AI(B, N_fixed, d_fixed) for B in B_values] #Corresponding decodes AIs for B_values
#Plot the AI Vs B graph for decode.
plt.figure(figsize=(8, 6))
plt.plot(B_values, decode_ai_values, marker='o')
plt.title("Arithmetic Intensity (AI) vs Batch Size (B) for Decode Attention")
plt.xlabel("Batch Size (B)")
plt.ylabel("Arithmetic Intensity (FLOPs/Bytes)")
plt.savefig("AI_vs_B_DecodeAttention.png", dpi=300)
plt.grid(True)
plt.show()
