#include <cuda_runtime.h>
#include <device_launch_parameter.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <random>

// Function to generate adjacency matrix
int init_adjacency_matrix(int *adj, int seqLength, int embDimension, int sparcity)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int nnz = 0;
    // Generate and print the matrix
    for (int i = 0; i < seqLength; ++i)
    {
        for (int j = 0; j < embDimension; ++j)
        {
            // Generate a random number and compare with sparsity
            adj[i * embDimension + j] = dis(gen) < sparsity ? 1 : 0;
            if (adj[i * embDimension + j] == 1)
            {
                nnz++;
            }
        }
    }
    return nnz;
}

// Function for generating random matrix using Xavier(Glorot) initialization

// Generate random matrix of values between 0 to 99
void init_matrix(int *mat, int seqLength, int embDimension)
{
    for (int i = 0; i < seqLength; i++)
    {
        for (int j = 0; j < embDimension; j++)
        {
            mat[i * embDimension + j] = rand() % 100;
        }
    }
}

// Kernels
// 1st matrix multiplication
__global__ void QxK_and_exp(float *Q, float *K, int seqLength, int embDimension, int *row, int *col, float *w, int nopt, float *exp_sum)
{
    // Identifying the corresponding part from the coo
    int initial = ((blockIdx.x * blockDim.x) + threadIdx.x) * nopt;
    int r, c;
    for (int i = initial; i < (initial + nopt); i++)
    {
        // dot product of corresponding row and col in coo - can be done using cuda
        r = row[i];
        c = col[i];
        int sum = 0;
        for (int j = 0; j < embDimension; j++)
        {
            sum += Q[r][j] * K[j][c];
        }
        w[i] = exp(sum);
        exp_sum[r] += w[i];
    }
}

// Divide the values in w matrix with exp_sum
__global__ void sum_div(float *w, float *exp_sum, int *row, int nopt)
{
    // Identifying the corresponding part from the coo
    int initial = ((blockIdx.x * blockDim.x) + threadIdx.x) * nopt;
    for (int i = initial; i < (initial + nopt); i++)
    {
        w[i] /= exp_sum[row[i]];
    }
}

// wxV  (method 1)

// ----------------------
// I have <<<blocksize, gridsize>>> computer power.
// my matrix size is greater than the blocksize x gridsize. I want to perform multiplication in following ways.
// 1. I consider a variable ncpb(number of computations per block - it will actually calculated by dividing (seqLength*embDimension) by (gridsize).
// 2. Now, every block will perform computation of ncpb consecutive data points in result matrix
// 3. each block does multiplications iteratively over all it's corresponding data points.
// 4. for computing them it will follow the following approach
//      a. first all the threads perform multiplications corresponding to a datapoint and store it in some intermediate memory
//      b.now a thread will add 2 consecutive values and store it back. this will go on.
//      It does computation of datapoints in O(log(n)) compared to the original one which does in O(n).
// ---------------------------------

// Kernel for sparse matrix (COO) * dense matrix multiplication with reduction
__global__ void wxV(int *d_row, int *d_col, float *d_w, int nnz, float *d_V, int seqLength, int embDimension, float *d_result, int ncpb)
{

    extern __shared__ float sharedMemory[]; // Dynamically allocated shared memory

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockStartIdx = blockIdx.x * ncpb; // Start index of the block's responsibility

    // Iterate over ncpb elements for this block
    for (int i = 0; i < ncpb; ++i)
    {
        int idx = blockStartIdx + i;

        // Ensure we are within the valid range
        if (idx < seqLength * embDimension)
        {
            int row = idx / embDimension;
            int col = idx % embDimension;

            float value = 0.0f;

            // Each thread performs the multiplication for the specific element
            for (int j = 0; j < nnz; ++j)
            {
                if (d_row[j] == row)
                {
                    value += d_w[j] * d_V[d_col[j] * embDimension + col];
                }
            }

            // Store result in shared memory
            sharedMemory[threadIdx.x] = value;
            __syncthreads();

            // Perform reduction in O(log(n))
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
            {
                if (threadIdx.x < stride)
                {
                    sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + stride];
                }
                __syncthreads();
            }

            // The final result is in sharedMemory[0] of the block
            if (threadIdx.x == 0)
            {
                atomicAdd(&d_result[idx], sharedMemory[0]); // Write result to global memory
            }
        }
    }
}

// // wxV (method 2)
// // Kernel for performing sparse matrix (in COO format) * dense matrix multiplication
// __global__ void wxV(float* d_w, int* d_row, int* d_col, float* d_V, int seqLength, int embDimension, float* d_result, int nopt) {
//     // Row and column of the non-zero element in sparse matrix
//     int idx = (blockIdx.x * blockDim.x + threadIdx.x)*nopt;

//     // Sparse matrix element exists at row d_row[idx] and col d_col[idx]
//     for  (int i = idx; i < idx+nopt; i++) {
//         if (idx < seqLength) {
//             int row = d_row[i];
//             int col = d_col[i];
//             float w_value = d_w[i];

//             // Perform the matrix multiplication: multiply sparse matrix value with each element in the dense matrix
//             for (int j = 0; j < embDimension; ++j) {
//                 atomicAdd(&d_result[row * embDimension + j], w_value * d_V[col * embDimension + j]);
//             }
//         }
//     }
// }

int main()
{
    // size of matrices
    int seqLength = 1 << 4;    // Sequence length
    int embDimension = 1 << 4; // Embedding dimension

    // Sparcity of adjacency matrix - Sparcity of the Attention
    int sparcity = 0.5;

    // define size  of matrices - for memory allocation
    size_t matrixSize = sizeof(float) * seqLenght * embDimension;     // for Q K V and result matrices
    size_t attentionMatrixSize = sizeof(int) * seqLength * seqLenght; // for adj
    size_t sumOfVectorsSize = sizeof(float) * seqLength;              // for sumOfVectors

    // Note: prefix h for a variable represents that the momory allocation is on host and prefix d for a variable represents that the memory allocation is on device

    // Initialize host pointers
    float *h_Q, *h_K, *h_V, *h_sumOfVectors, *h_result;
    int *h_attentionMatrix;

    // Allocate memory at the host
    h_Q = (*float)malloc(matrixSize);
    h_K = (*float)malloc(matrixSize);
    h_V = (*float)malloc(matrixSize);
    h_attentionMatrix = (*int)malloc(attentionMatrixSize);
    h_sumOfVectors = (*float)malloc(sumOfVectorsSize);
    h_result = (*float)malloc(matrixSize);

    // The memory allocation for attentionMatrix on device is not needed as the coo matrix is being computed on host

    // Initialize device pointers
    float *d_Q, *d_K, *d_V, *d_sumOfVectors, *d_result;

    // allocate memory on the device
    cudaMalloc(&d_Q, matrixSize);
    cudaMalloc(&d_K, matrixSize);
    cudaMalloc(&d_V, matrixSize);
    cudaMalloc(&d_sumOfVectors, sumOfVectorsSize);
    cudaMalloc(&d_result, matrixSize);

    // Initialize matrices with random values
    init_matrix(h_Q, seqLength, embDimension);
    init_matrix(h_K, seqLength, embDimension);
    init_matrix(h_V, seqLength, embDimension);
    int nnz = init_adjacency_matrix(h_adj, seqLength, embDimension, sparcity);

    // copy memory to the device
    cudaMemcpy(d_Q, h_Q, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, matrixSize, cudaMemcpyHostToDevice);

    // represent adjacency matrix in coo form
    int *h_row, *h_col;
    float *h_w; // for storing intermediate result
    size_t nonZeroInt = nnz * sizeof(int);
    size_t nonZeroFloat = nnz * sizeof(float);
    h_row = (*int)malloc(nonZeroInt);
    h_col = (*int)malloc(nonZeroInt);
    h_w = (*float)malloc(nonZeroFloat);

    int *d_row, *d_col;
    float *d_w;
    cudaMalloc(&d_row, nonZeroInt);
    cudaMalloc(&d_col, nonZeroInr);
    cudaMalloc(&d_w, nonZeroFloat);

    // Generate COO
    int cooIndex = 0; // index for coo
    for (int i = 0; i < seqLength; i++)
    {
        for (int j = 0; j < embDimension; j++)
        {
            if (h_adj[i * embDimension + j] == 1)
            {
                h_row[cooIndex] = i;
                h_col[cooIndex] = j;
                cooIndex++; // increment after each non-zero element
            }
        }
    }

    free(h_adj);

    // transfer coo representation of adj
    cudaMemcpy(d_row, h_row, nonZeroInt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col, nonZeroInt, cudaMemcpyHostToDevice);

    // Defining required GPU architecture

    // Define BlockSize
    int BlockSize = 256;

    // Calculate gridSize
    int GridSize = 16;

    // Number of operations per thread
    int nopt = ceil(nnz / (BlockSize * GridSize));

    // Initialize allocated memory to zero
    memset(h_w, 0, nonZeroFloat);
    cudaMemcpy(d_w, h_w, nonZeroFloat, cudaMemcpyHostToDevice);

    // Initialize h_sum to 0 and copy that to device meomory
    memset(h_sumOfVectors, 0, sumOfVectorsSize);
    cudaMemcpy(d_sumOfVectors, h_sumOfVectors, sumOfVectorsSize, cudaMemcpyHostToDevice);

    // Call the Kernel (for first matrix multiplication and also exponential and addition of values of corresponding rows)
    QxK_and_exp<<<GridSize, BlockSize>>>(d_Q, d_K, seqLength, embDimension, d_row, d_col, d_w, nopt, d_sumOfVectors);

    cudaDeviceSynchronize();

    // call the kernel (for division by sum of exponentials)
    sum_div<<<GridSize, BlockSize>>>(d_w, d_sumOfVectors, d_row, nopt);

    cudaDeviceSynchronize();

    // Initialize result matrix to zero and pass it to device memory
    memset(h_result, 0, matrixSize);
    cudaMemcpy(d_result, h_result, matrixSize, cudaMemcpyHostToDevice);

    // call the kernel (for final matrix multiplication) - method 1
    int ncpt = ceil((seqLength * embDimension) / GridSize); // number of computations per block
    wxV<<<GridSize, BlockSize>>>(d_row, d_col, d_w, nnz, d_V, seqLength, embDimension, d_result, ncpt);

    // // call the kernel (for final matrix multiplication) - method 2
    // wxV<<<GridSize, BlockSize>>>(d_w, d_row, d_col, d_V, seqLength, embDimension, d_result, nopt);

    cudaDeviceSynchronize();

    // free device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_sumOfVectors);
    cudaFree(d_col);
    cudaFree(d_row);

    // free host memory
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_sumOfVectors);
    free(h_col);
    free(h_row);
}
