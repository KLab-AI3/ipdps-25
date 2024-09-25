import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import spfa_csr

import math
import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

# Store the specified dimensions.
Q = 10
K = 10
V = 10
D = 10

# Specify the level density of the mask (1 -> fully dense, 0 -> fully sparse).
DENSE_LEVEL = 1

# Set the default device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Control the data type of individual tensors.
data_type = torch.float32

# Verify that the default data type for tensor creation is float32 (single-precision floating-point).
torch.set_default_dtype(torch.float32)

# Create random tensors of the specified dimensions.
q = torch.rand([Q, D], device = device, dtype = data_type)
k = torch.rand([K, D], device = device, dtype = data_type)
v = torch.rand([V, D], device = device, dtype = data_type)

# Create random connection of the provided sparsity level for the mask.
mask = torch.rand((Q, Q), device = device, dtype = data_type) < DENSE_LEVEL

# Calculate the PyTorch attention result.
# with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
with sdpa_kernel(SDPBackend.MATH):
# with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    torch_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
    torch.cuda.synchronize()

# NOTE: Restrict the mask to int32
mask = mask.type(data_type)
csr_mask = mask.to_sparse_csr()
w_row_off = csr_mask.crow_indices().type(torch.uint64)
w_col_ind = csr_mask.col_indices().type(torch.uint32)
w_val = csr_mask.values()

o = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.zeros(Q, device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

o = spfa_csr.forward(q, k, v, w_row_off, w_col_ind, w_val, m, l, o)
torch.cuda.synchronize()

# Verify that the PyTorch SDPA's outputs are identical
if torch.allclose(torch_out, o, atol = 1e-3):
    print("The SPFA-CSR results match PyTorch's.")

print("Max difference (SPFA-CSR):", torch.max(o - torch_out))
print("Min difference (SPFA-CSR):", torch.min(o - torch_out))

print("Finished")