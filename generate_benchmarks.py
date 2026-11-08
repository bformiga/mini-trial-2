import torch
import numpy as np

def sparsity_2_4(dense_weights):
    K, N = dense_weights.shape
    
    # 1. Break it up into blocks of 4
    reshaped = dense_weights.reshape(K, N // 4, 4)
    
    # 2. Find the 2 largest weights by absolute value
    abs_weights = torch.abs(reshaped)
    _, keep_indices = torch.topk(abs_weights, k=2, dim=-1)
    mask = torch.zeros_like(reshaped, dtype=torch.bool)
    mask.scatter_(dim=-1, index=keep_indices, value=True)
    
    # 3. Apply the mask and reshape back to (K, N)
    sparse_weights = (reshaped * mask).reshape(K, N)
        
    return sparse_weights

def generate_benchmark():
    # The workloads we are using + the padding. It goes M, N, K
    workloads = {
        "resnet50_conv1": [12544, 64, 160],
        "llama3_q_proj":  [32, 1024, 4096],
        "llama3_gate_proj":[256, 1024, 4096]
    }
    
    # Looping through the 3 workloads
    for name, dim in workloads.items():
        M, N, K = dim
        
        # Initializing matrices for the activation and dense
        activation = torch.randn(M, K, dtype=torch.float16)
        dense_weights = torch.randn(K, N, dtype=torch.float16)
        
        # Applying sparsity
        sparse_weights = sparsity_2_4(dense_weights)
        
        # Outputs of the dense and sparse multiplication
        dense_output = torch.matmul(activation, dense_weights)
        sparse_output = torch.matmul(activation, sparse_weights)
        
        # Saving the data for each matrix
        files_to_save = {
            f"{name}_activations.log": activation,
            f"{name}_weights_dense.log": dense_weights,
            f"{name}_weights_sparse.log": sparse_weights,
            f"{name}_output_dense.log": dense_output,
            f"{name}_output_sparse.log": sparse_output
        }
        for filename, file_type in files_to_save.items():
            np.save(filename, file_type.numpy().astype(np.float16))

if __name__ == "__main__":
    torch.manual_seed(100) 
    generate_benchmark()