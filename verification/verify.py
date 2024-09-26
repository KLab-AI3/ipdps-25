import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import spfa_csr
import spfa_coo

import math
import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

# Store the specified dimensions.
Q = 256
K = 256
V = 256
D = 32

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

# Clone the mask.
mask_2 = torch.clone(mask)
mask_3 = torch.clone(mask)

# Convert the PyTorch mask to float to use 
# mask = mask.type(torch.float32)
# mask = torch.where(mask == 1.0, 0.0, -float("inf"))

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

o_csr = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.zeros(Q, device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

o_csr = spfa_csr.forward(q, k, v, w_row_off, w_col_ind, w_val, m, l, o_csr)
torch.cuda.synchronize()

# Verify that the PyTorch SDPA's outputs are identical
if torch.allclose(torch_out, o_csr):
    print("The SPFA-CSR results match PyTorch's.")

print("Max difference from PyTorch (SPFA-CSR):", torch.max(o_csr - torch_out))
print("Min difference from PyTorch (SPFA-CSR):", torch.min(o_csr - torch_out))

# NOTE: Restrict the mask to int32
mask_3 = mask_3.type(data_type)
coo_mask = mask_3.to_sparse_coo()
w_row_ind = coo_mask.indices()[0].type(torch.uint64)
w_col_ind = coo_mask.indices()[1].type(torch.uint32)
w_val = coo_mask.values()

o_coo = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.zeros(Q, device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

o_coo = spfa_coo.forward(q, k, v, w_row_ind, w_col_ind, w_val, m, l, o_coo)
torch.cuda.synchronize()

# Verify that the PyTorch SDPA's outputs are identical
if torch.allclose(torch_out, o_coo):
    print("The SPFA-COO results match PyTorch's.")

print("Max difference from PyTorch (SPFA-COO):", torch.max(o_coo - torch_out))
print("Min difference from PyTorch (SPFA-COO):", torch.min(o_coo - torch_out))

# Verify that the CSR and COO outputs are identical
if torch.allclose(o_csr, o_coo):
    print("The SPFA-CSR results match SPFA-COO's.")

print("Finished")