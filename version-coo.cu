#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

#define NNZ 100       // Example number of non-zero elements
#define NUM_ROWS 10   // Example number of rows
#define NUM_COLS 10   // Example number of columns
#define NUM_THREADS 4 // Example number of threads

// Function to initialize matrices with random values
void initializeMatrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
    {
        matrix[i] = static_cast<float>(rand() % 10 + 1); // Random values between 1 and 10
    }
}

// Function to initialize COO format row and column arrays with random indices
void initializeCOO(int *row, int *col, int nnz, int rows, int cols)
{
    for (int i = 0; i < nnz; ++i)
    {
        row[i] = rand() % rows; // Random row index
        col[i] = rand() % cols; // Random column index
    }
}

// Function to initialize rowIndx array with indices for COO
void initializeRowIndx(int *rowIndx, int rows, int nnz)
{
    rowIndx[0] = 0;
    for (int i = 1; i <= rows; ++i)
    {
        rowIndx[i] = i * (nnz / rows); // Evenly distribute NNZ indices across rows
    }
    rowIndx[rows] = nnz; // Ensure the last index equals NNZ
}

// Kernel to perform the initial vector multiplication, exponential, and summation
__global__ void computeMatrixWAndSum(
    float *K, float *Q, float *W, float *V, int *row, int *col,
    int NNZ, int *sum, int numRows, int numCols, int p)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n = (NNZ + p - 1) / p; // Number of computations per thread

    int start = tid * n;
    int end = min(start + n, NNZ);

    for (int i = start; i < end; ++i)
    {
        int rowIndex = row[i];
        int colIndex = col[i];
        float val = 0.0f;

        // Perform vector multiplication Q[rowIndex] * K[colIndex]
        for (int j = 0; j < numCols; ++j)
        {
            val += Q[rowIndex * numCols + j] * K[colIndex * numCols + j];
        }

        float eval = exp(val);
        W[rowIndex * numCols + colIndex] = eval;
        atomicAdd(&sum[rowIndex], eval); // Atomic addition to sum array
    }
}

// Kernel to perform the addition and normalization
__global__ void computeFinalOutput(
    float *W, float *V, float *R, int *rowIndx, int *row, int *col,
    int numRows, int nnzPerRow)
{

    extern __shared__ float sharedW[];
    extern __shared__ float sharedVout[];

    int blockID = blockIdx.x;
    int tid = threadIdx.x;

    int BLOCK_SIZE = 256;
    int nnzPerThread = (nnzPerRow + BLOCK_SIZE - 1) / BLOCK_SIZE; // Number of non-zeros per thread
    int T = tid * nnzPerThread;
    int r = rowIndx[blockID] + T;

    for (int i = 0; i < nnzPerThread; ++i)
    {
        if (r + i < rowIndx[blockID + 1])
        { // Ensure we don't go out of bounds
            sharedW[tid * nnzPerThread + i] = W[r + i];
            sharedVout[tid * nnzPerThread + i] = V[r + i] * W[r + i];
        }
    }

    __syncthreads();

    // Phase I: Reduce the sum of W and Vout in log(nnzPerThread) steps
    for (int iter = 0; iter < log2f(nnzPerThread); ++iter)
    {
        int distance = 1 << iter;

        if (distance < nnzPerThread)
        {
            for (int j = 0; j + distance < nnzPerThread; j += distance)
            {
                sharedVout[T + j] += sharedVout[T + j + distance];
                sharedW[T + j] += sharedW[T + j + distance];
            }
        }

        __syncthreads();
    }

    // Phase II: Reduce within block for final output
    for (int iter = 0; iter < log2f(BLOCK_SIZE); ++iter)
    {
        int distance = 1 << iter;

        if (tid % distance == 0 && tid + distance < BLOCK_SIZE)
        {
            sharedVout[tid * nnzPerThread] += sharedVout[(tid + distance) * nnzPerThread];
            sharedW[tid * nnzPerThread] += sharedW[(tid + distance) * nnzPerThread];
        }

        __syncthreads();
    }

    // Final normalization for each block
    if (tid == 0)
    {
        R[blockID] = sharedVout[0] / sharedW[0];
    }
}

int main()
{
    // Allocate host memory
    float *K, *Q, *V, *Adj, *R;
    int *row, *col, *rowIndx;

    // Allocate matrices and arrays on the host
    K = (float *)malloc(NUM_ROWS * NUM_COLS * sizeof(float));
    Q = (float *)malloc(NUM_ROWS * NUM_COLS * sizeof(float));
    V = (float *)malloc(NUM_ROWS * NUM_COLS * sizeof(float));
    // Adj = (float *)malloc(NUM_ROWS * NUM_COLS * sizeof(float)); // If needed, use this for adjacency representation
    R = (float *)malloc(NUM_ROWS * NUM_COLS * sizeof(float));
    row = (int *)malloc(NNZ * sizeof(int));
    col = (int *)malloc(NNZ * sizeof(int));
    rowIndx = (int *)malloc((NUM_ROWS + 1) * sizeof(int));

    // Initialize matrices with random values
    initializeMatrix(K, NUM_ROWS, NUM_COLS);
    initializeMatrix(Q, NUM_ROWS, NUM_COLS);
    initializeMatrix(V, NUM_ROWS, NUM_COLS);
    // initializeMatrix(Adj, NUM_ROWS, NUM_COLS); // Initialize if necessary for other purposes

    // Initialize COO format arrays
    initializeCOO(row, col, NNZ, NUM_ROWS, NUM_COLS);

    // Initialize rowIndx array
    initializeRowIndx(rowIndx, NUM_ROWS, NNZ);

    // Allocate device memory
    float *d_K, *d_Q, *d_V, *d_W, *d_R;
    int *d_row, *d_col, *d_rowIndx, *d_sum;
    cudaMalloc((void **)&d_K, NUM_ROWS * NUM_COLS * sizeof(float));
    cudaMalloc((void **)&d_Q, NUM_ROWS * NUM_COLS * sizeof(float));
    cudaMalloc((void **)&d_V, NUM_ROWS * NUM_COLS * sizeof(float));
    cudaMalloc((void **)&d_W, NUM_ROWS * NUM_COLS * sizeof(float));
    cudaMalloc((void **)&d_R, NUM_ROWS * NUM_COLS * sizeof(float));
    cudaMalloc((void **)&d_row, NNZ * sizeof(int));
    cudaMalloc((void **)&d_col, NNZ * sizeof(int));
    cudaMalloc((void **)&d_rowIndx, (NUM_ROWS + 1) * sizeof(int));
    cudaMalloc((void **)&d_sum, NUM_ROWS * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_K, K, NUM_ROWS * NUM_COLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Q, NUM_ROWS * NUM_COLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, NUM_ROWS * NUM_COLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, row, NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col, NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowIndx, rowIndx, (NUM_ROWS + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, sum, NUM_ROWS * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim((NNZ + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    // Launch kernels
    computeMatrixWAndSum<<<gridDim, blockDim>>>(
        d_K, d_Q, d_W, d_V, d_row, d_col, NNZ, d_sum, NUM_ROWS, NUM_COLS, NUM_THREADS);

    // Use a shared memory size that fits nnzPerThread of W and Vout
    int nnzPerRow = NNZ / NUM_THREADS;
    computeFinalOutput<<<NUM_ROWS, blockDim, nnzPerRow * sizeof(float)>>>(
        d_W, d_V, d_R, d_rowIndx, d_row, d_col, NUM_ROWS, nnzPerRow);

    // Copy the result back to host
    cudaMemcpy(R, d_R, NUM_ROWS * NUM_COLS * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results (for debugging)
    for (int i = 0; i < NUM_ROWS; ++i)
    {
        std::cout << R[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_V);
    cudaFree(d_W);
    cudaFree(d_R);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_rowIndx);
    cudaFree(d_sum);

    // Free host memory
    free(K);
    free(Q);
    free(V);
    // free(Adj);
    free(R);
    free(row);
    free(col);
    free(rowIndx);

    return 0;
}
