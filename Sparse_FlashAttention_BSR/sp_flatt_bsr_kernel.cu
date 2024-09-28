#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <stdio.h>

#include <torch/extension.h>


// The sparse-FlashAttention (BSR mask) forward CUDA kernel.
template <typename scalar_t>
__global__ void spfa_bsr_cuda_forward_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> K,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> V,
    const torch::PackedTensorAccessor64<unsigned long int, 1, torch::RestrictPtrTraits> W_block_row_off,
    const torch::PackedTensorAccessor64<unsigned int, 1, torch::RestrictPtrTraits> W_block_col_ind,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> W_val,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> m,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> l,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> O,
    int block_length,
    int maskElePerIter
) {
  // Keep track of the lower and upper bound indices of the W block column index vector.
  int lowInd = W_block_row_off[blockIdx.x];
  int upInd = W_block_row_off[blockIdx.x + 1];

  // Calculate the total number of non-masked elements in the row of the mask.
  int totNonMask = block_length * (upInd - lowInd);

  // Track the number of entries that have been calculated. This will help us figure out which block and item we are working 
  // on if more than one iteration is required to calculate the row of O.
  int calced_tracker = 0;

  // Calculate the number of times the amount of threads will fit into the embedded dimension as well as the remainder.
  int intQuotient_ED = (int)(Q.size(1) / blockDim.x);
  int remainder_ED = (Q.size(1) % blockDim.x) - 1;

  // If the thread index is less than the remainder, then increment its quotient so it reads one more value.
  if (threadIdx.x <= remainder_ED) {
    intQuotient_ED += 1;
  }

  // If the row is fully masked, then fill the output with NaN's and end the iteration.
  // NOTE: Check if this is correct or if I should set it to another value.
  if (totNonMask == 0) {
    for (int i = 0; i < intQuotient_ED; i++) {
      O[blockIdx.x][(blockDim.x * i) + threadIdx.x] = (scalar_t)NAN;
      // __syncthreads();
    }

    return;
  }

  // If the total number of non-masked elements is under the max number of mask elements per iteration, 
  // then reduce the max number of elements for this iteration.
  if (totNonMask < maskElePerIter) {
    maskElePerIter = totNonMask;
  }

  // Create the shared memory object, aligned to the size of the template data type.
  extern __shared__ __align__(sizeof(scalar_t)) unsigned char sharedMem[];

  // Get pointers to the shared memory and reinterpret the pointer's data type.
  scalar_t* Q_shared = reinterpret_cast<scalar_t*>(sharedMem);
  scalar_t* O_shared = (scalar_t*)&Q_shared[Q.size(1)];
  scalar_t* W_val_shared = (scalar_t*)&O_shared[O.size(1)];
  scalar_t* K_shared = (scalar_t*)&W_val_shared[maskElePerIter];
  scalar_t* V_shared = (scalar_t*)&K_shared[maskElePerIter * K.size(1)];
  scalar_t* m_i = (scalar_t*)&V_shared[maskElePerIter * V.size(1)];
  scalar_t* m_ij = (scalar_t*)&m_i[1];
  scalar_t* m_new_i = (scalar_t*)&m_ij[1];
  scalar_t* l_i = (scalar_t*)&m_new_i[1];
  scalar_t* l_ij = (scalar_t*)&l_i[1];
  scalar_t* l_new_i = (scalar_t*)&l_ij[1];

  // Bring Q and O into shared memory.
  for (int i = 0; i < intQuotient_ED; i++) {
    Q_shared[(blockDim.x * i) + threadIdx.x] = Q[blockIdx.x][(blockDim.x * i) + threadIdx.x];
    O_shared[(blockDim.x * i) + threadIdx.x] = O[blockIdx.x][(blockDim.x * i) + threadIdx.x];
    // __syncthreads();
  }

  // Move the softmax statistics into shared memory.
  //// - m is initialized with 0.0.
  //// - l is initialized with -inf.
  //// - If they are set to something else in shared memory, then those values will be used.
  if (threadIdx.x == 0) {
    m_i[0] = m[blockIdx.x];
    l_i[0] = l[blockIdx.x];
  }

  // Block until all threads are synchronized after the last operation.
  __syncthreads();

  // Calculate the number of iterations that the block will have to perform.
  int blockIters = ceil(((float)totNonMask / (float)maskElePerIter));

  // Calculate the number of masked terms that remain on the last iteration.
  int numLastIter = (totNonMask % maskElePerIter);

  // The main calculation loop.
  for (int b_i = 0; b_i < blockIters; b_i++) {
    // If we're on the last iteration and it's not the first, then set the number of masked elements equal to those 
    // that have not been visited yet.
    if ((b_i == (blockIters - 1)) && (b_i != 0)) {
      maskElePerIter = numLastIter;
    }

    // Calculate the column indices and bring K and V into shared memory.
    //// - W's value vector is overwritten the first iteration, so there is no need to read from HBM or initialize it to something.
    //// - K_shared is contiguous along the rows of K.
    //// - V_shared is contiguous along the columns of V.
    if (threadIdx.x < maskElePerIter) {
      // Find where the thread's position is within the total number of calculations that must be conducted.
      int position = calced_tracker + threadIdx.x;

      // Find which block index the thread is associated with and adjust it using the lower bound index pointer.
      int block_ind = floor(position / block_length) + lowInd;

      // Find the thread's position within said block.
      int block_in_ind = position % block_length;

      // Calculate and store the column index of the mask so we don't have to do it each iteration below.
      int col_ind = (block_length * W_block_col_ind[block_ind]) + block_in_ind;

      for (int i = 0; i < K.size(1); i++) {
        K_shared[(K.size(1) * threadIdx.x) + i] = K[col_ind][i];
        V_shared[(maskElePerIter * i) + threadIdx.x] = V[col_ind][i];
      }
    }

    // Update the tracker to know how many elements of the mask have been operated on thus far.
    calced_tracker += maskElePerIter;

    __syncthreads();

    if (threadIdx.x < maskElePerIter) {
      scalar_t acc = 0.0;

      for (int i = 0; i < Q.size(1); i++) {
        acc += Q_shared[i] * K_shared[(Q.size(1) * threadIdx.x) + i];
      }

      W_val_shared[threadIdx.x] = acc / sqrt((float)Q.size(1));
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      scalar_t row_max = - (1.0 / 0.0);

      for (int i = 0; i < maskElePerIter; i++) {
        if (W_val_shared[i] > row_max) {
          row_max = W_val_shared[i];
        }
      }

      m_ij[0] = row_max;
    }

    __syncthreads();

    if (threadIdx.x < maskElePerIter) {
      W_val_shared[threadIdx.x] = exp(W_val_shared[threadIdx.x] - m_ij[0]);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      scalar_t row_sum = 0.0;

      for (int i = 0; i < maskElePerIter; i++) {
        row_sum += W_val_shared[i];
      }

      l_ij[0] = row_sum;

      if (m_i[0] > m_ij[0]) {
        m_new_i[0] = m_i[0];
      }
      else {
        m_new_i[0] = m_ij[0];
      }

      l_new_i[0] = (l_i[0] * exp(m_i[0] - m_new_i[0])) + (l_ij[0] * exp(m_ij[0] - m_new_i[0]));
    }

    __syncthreads();

    // NOTE: This does not work for sizes greater than number of threads.
    if (threadIdx.x < V.size(1)) {
      scalar_t acc = 0.0;

      for (int i = 0; i < maskElePerIter; i++) {
        acc += W_val_shared[i] * V_shared[(threadIdx.x * maskElePerIter) + i];
      }

      O_shared[threadIdx.x] = (1.0 / l_new_i[0]) * 
        ((l_i[0] * O_shared[threadIdx.x] * exp(m_i[0] - m_new_i[0])) + 
        (acc * exp(m_ij[0] - m_new_i[0])));
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      m_i[0] = m_new_i[0];
      l_i[0] = l_new_i[0];
    }

    __syncthreads();

  }

  // Update the softmax statistics in HBM.
  if (threadIdx.x == 0) {
      m[blockIdx.x] = m_new_i[0];
      l[blockIdx.x] = l_new_i[0];
  }

  __syncthreads();

  for (int i = 0; i < intQuotient_ED; i++) {
    O[blockIdx.x][(blockDim.x * i) + threadIdx.x] = O_shared[(blockDim.x * i) + threadIdx.x];
    // __syncthreads();
  }
  
}


// The CUDA kernel dispatch interface for the sparse-FlashAttention (BSR mask) forward CUDA kernel.
torch::Tensor spfa_bsr_cuda_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor W_block_row_off,
    torch::Tensor W_block_col_ind,
    torch::Tensor W_val,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor O,
    int block_length
) {
  // Initialize the variables to hold the desired device attributes.
  //// - Max number of threads per block.
  //// - Max shared memory per block (bytes).
  int maxThrPB, maxSMemPB;

  // Get the device attributes.
  cudaDeviceGetAttribute(&maxThrPB, cudaDevAttrMaxThreadsPerBlock, 0);
  cudaDeviceGetAttribute(&maxSMemPB, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  // Set the grid dimensions for the problem.
  //// - The number of threads is set to the max per block for the device.
  //// - The number of blocks is equal to the number of rows in Q.
  const int threads = maxThrPB;
  const dim3 blocks(Q.size(0));

  // Store the embedded dimension.
  int embD = Q.size(1);

  // Calculate the max number of mask elements that will be utilized for each iteration of building O.
  //// - maxSMemPB -> The amount of shared memory available to a block in bytes.
  //// - (2 * embD * sizeof(Q.type())) -> (numerator) The amount of shared memory that storing Q and O will require.
  //// - (6 * sizeof(l.type())) -> The amount of shared memory that storing the softmax statistics will require.
  //// - sizeof(W_val.type()) -> The amount of shared memory that storing the mask's values will require.
  //// - (2 * embD * sizeof(Q.type())) -> (denominator) The amount of shared memory that storing K and V will require.
  int maskElePerIter = floor((maxSMemPB - (2 * embD * sizeof(Q.type())) - (6 * sizeof(l.type()))) / 
    (sizeof(W_val.type()) + (2 * embD * sizeof(Q.type()))));

  // If the number of mask elements per iteration is less than 1, then throw an error.
  if (maskElePerIter < 1) {
    throw std::invalid_argument( "The embedded dimension is too large for the amount of shared memory on your GPU." );
  }

  // Dispatcher that handles launching the correctly typed function from the generic implementation.
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.type(), "spfa_bsr_forward_cuda", ([&] {
    spfa_bsr_cuda_forward_kernel<scalar_t><<<blocks, 256, maxSMemPB>>>(
        Q.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        K.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        V.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        W_block_row_off.packed_accessor64<unsigned long int, 1, torch::RestrictPtrTraits>(),
        W_block_col_ind.packed_accessor64<unsigned int, 1, torch::RestrictPtrTraits>(),
        W_val.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
        m.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
        l.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
        O.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        block_length,
        maskElePerIter
      );
  }));

  // Return the output tensor.
  return O;
}
