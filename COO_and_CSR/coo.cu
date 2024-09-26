#include <cuda_runtime.h>
#include <device_launch_parameter.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <random>

// Function to generate adjacency matrix
int init_adjacency_matrix(int *adj, int n, int sparcity)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int nnz = 0;
    // Generate and print the matrix
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            // Generate a random number and compare with sparsity
            adj[i * n + j] = dis(gen) < sparsity ? 1 : 0;
            if (adj[i * n + j] == 1)
            {
                nnz++;
            }
        }
    }
    return nnz;
}

// Generate random matrix of values between 0 to 99
void init_matrix(int *mat, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mat[i * n + j] = rand() % 100;
        }
    }
}

// Kernels
// 1st matrix multiplication
__global__ void QxK_and_exp(float *Q, float *K, int *row, int *col, float *w, int nopt, float *exp_sum)
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
        for (int j = 0; j < n; j++)
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

// final matrix multiplication
//  __global__ void wxV(float *w, float *V, int *row, int *col, int n, int nopt, float *result){
//      int initial = ((blockIdx.x * blockDim.x) + threadIdx.x ) * nopt;
//      for (int i=initial; i<(initial+nopt) ; i++){
//          // dot product of corresponding row and col in coo - can be done using cuda
//          r=row[i];
//          c=col[i];
//          for(int j=0;j<n;j++){
//              res[r*n+j]+=w[r*n+c]*V[c*n+j];
//          }
//      }
//      __syncthreads();
//  }

// Final matrix multiplication

__global__ void wxV(float *w, float *V, int *row, int *col, int n, float *result)
{

    vector<int> r;
    r.pushback(0);
    int c = 0;
    int cnum = col[0];
    for (int i = 0; i < nnz; i++)
    {
        if (col[i] == cnum)
        {
            c++;
        }
        else
        {
            r.push_back(cnum);
            c = 0;
        }
    }

    int i = blockIdx / n; // row index
    int j = blockIdx % n; // column index
    int idx = i * n + j;

    float sum = 0.0f;
    int start = row[i];
    int end = row[i + 1];

    for (int k = start; k < end; k++)
    {
        int col_idx = col[k];
        sum += w[k] * V[col_idx * n + j];
    }

    result[idx] = sum;
}

int main()
{
    // size of matrices
    int n = 1 << 4;

    // Sparcity of adjacency matrix
    int sparcity = 0.5;

    // define size
    size_t bytes = sizeof(float) * n * n;

    // Initialize host pointers
    float *h_Q, *h_K, *h_V, *h_sum, *h_result;
    int *h_adj;

    // Allocate memory at the host
    h_Q = (*float)malloc(bytes);
    h_K = (*float)malloc(bytes);
    h_V = (*float)malloc(bytes);
    h_adj = (*int)malloc(n * sizeof(int));
    h_sum = (*float)malloc(n * sizeof(float));
    h_result = (*float)malloc(bytes);

    // Initialize device pointers
    float *d_Q, *d_K, *d_V, *d_sum, *d_result;

    // allocate memory on the device
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_sum, n * sizeof(float));
    cudaMalloc(&d_result, bytes);

    // Initialize matrices with random values
    init_matrix(h_Q, n);
    init_matrix(h_K, n);
    init_matrix(h_V, n);
    int nnz = init_adjacency_matrix(h_adj, n, sparcity);

    // copy memory to the device
    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

    // represent adjacency matrix in coo form
    int *h_row, *h_col;
    float *h_w;
    size_t nz = nnz * sizeof(int);
    h_row = (*int)malloc(nz);
    h_col = (*int)malloc(nz);
    h_w = (*float)malloc(nnz * sizeof(float));

    int *d_row, *d_col;
    float *d_w
        cudaMalloc(&d_row, nz);
    cudaMalloc(&d_col, nz);
    cudaMalloc(&d_w, nnz * sizeof(float));
    int k = 0; // index for coo
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (h_adj[i * n + j] == 1)
            {
                h_row[nnz++] = i;
                h_col[nnz++] = j;
            }
        }
    }

    // transfer coo representation of adj
    cudaMemcpy(d_row, h_row, nz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col, nz, cudaMemcpyHostToDevice);

    // Define BlockSize
    int BlockSize = 4;

    // Calculate gridSize
    int GridSize = 2;

    // Number of operations per thread
    int nopt = ceil(nnz / (BlockSize * GridSize));

    // Initialize allocated memory to zero
    memset(h_w, 0, bytes);
    cudaMemcpy(d_w, h_w, bytes, cudaMemcpyHostToDevice);

    // Initialize h_sum to 0 and copy that to device meomory
    memset(h_sum, 0, n * sizeof(float));
    cudaMemcpy(d_sum, h_sum, n * sizeof(float), cudaMemcpyHostToDevice);

    // Call the Kernel (for multiplication and exponential and addition of values)
    QxK_and_exp<<<GridSize, BlockSize>>>(d_Q, d_K, d_row, d_col, d_w, nopt, d_sum);

    cudaDeviceSynchronize();

    // division by sum of exponentials
    sum_div<<<GridSize, BlockSize>>>(d_w, d_sum, d_row, nopt);

    cudaDeviceSynchronize();

    // Initialize result matrix to zero and pass it to device memory
    memset(h_result, 0, bytes);
    cudaMemcpy(d_result, h_result, bytes, cudaMemcpyHostToDevice);

    // call the kernel (for final matrix multiplication)
    wxV<<<GridSize, BlockSize>>>(d_w, d_V, d_row, d_col, n, nopt, d_result);

    cudaDeviceSynchronize();

    // free device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_sum);
    cudaFree(d_col);
    cudaFree(d_row);

    // free host memory
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_sum);
    free(h_col);
    free(h_row);
}
