#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <mma.h>

#include "sparse_utilities.h"
#include "comp_mats.h"

// For explicitation the defined WARP_SIZE is used instead of WARP_SIZE
#define WARP_SIZE 32

// These must be multiples of 16 for WMMA to work
// These are the only dimensions currently supported by WMMA
// Other values will throw CUDA error: incomplete type is not allowed
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Most trivial example: M = 16, N = 16, K = 16
// Define the GEMM Tile size
#define M_TILES 16 * 16
#define N_TILES 16 * 16
#define K_TILES 16 * 16

// In case of odd numbers -> CUDA error: misaligned address
// Turing Tensor Cores also support tiles sizes 32 * 8 * 16, 8 * 32 * 16
#define M_GLOBAL (WMMA_M * 4)
#define N_GLOBAL (WMMA_N * 4)
#define K_GLOBAL (WMMA_K * 4)

// This is used for padding inside the shared memory
#define SKEW_K 16

// Custom error checking macro;
#define cudaCheck(stat)                         \
	{                                           \
		cudaCheck_((stat), __FILE__, __LINE__); \
	}
void cudaCheck_(cudaError_t stat, const char *file, int line)
{
	if (stat != cudaSuccess)
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
}

// Define the context of functions and types for the nvcuda namespace
using namespace nvcuda;

// Simple kernel for wmma sparse-dense multiplication
__global__ void wmma_test_simple(const half *__restrict__ A, const half *__restrict__ B, float *__restrict__ C, const int M, const int N, const int K, const int R_BLOCKS, const int C_BLOCKS, const int NZ_BLOCKS, const int *NZ_COUNT, const int *VBS_JAB, const int NZ_ELEMENTS)
{
	// Leading dimensions
	int lead_A = M;
	int lead_B = K;
	int lead_C = M;

	// 2D tile division in warps 
	int warp_M = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
	int warp_N = (blockIdx.y * blockDim.y + threadIdx.y);
	
	// printf("Thread = %d, warp_M = %d, warp_K = %d\n", threadIdx.x, warp_M, warp_K);

	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_A;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_B;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_C;

	// Initialize the output to zero
	wmma::fill_fragment(frag_C, 0.0f);

	// Number of non-zero block in each row (warp_M dimension)
	const int nzblocks_R = NZ_COUNT[warp_M];
	
	// Outer K loop (in blocks)
	if (nzblocks_R > 0)
	{
		// Row of blocks 
		for (int i = 0; i < nzblocks_R; ++i)
		{  
			const int block_R = VBS_JAB[warp_M];
			const int idx_R = warp_M * M_TILES;

			const int block_C = VBS_JAB[warp_N];
			const int idx_C = block_C * WMMA_N;

			// Bounds checking
			if (warp_M < R_BLOCKS && warp_K < C_BLOCKS && idx_R < M)
			{		
				wmma::load_matrix_sync(frag_A, A + block_R * M_TILES, WMMA_M);
				wmma::load_matrix_sync(frag_B, B, WMMA_N);

				// Execute matrix multiplication 
				wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

				// Store the result
				wmma::store_matrix_sync(C + idx_R * lead_C + idx_C, frag_C, lead_C, wmma::mem_row_major); 
			}
		}
	}

	int col_C = warp_N * WMMA_N;
	int row_C = warp_M * WMMA_M;

	// Bounds checking
	if (row_C < M && col_C < N) 
	{
    		wmma::load_matrix_sync(frag_C, C + col_C + row_C * lead_C, lead_C, wmma::mem_row_major);

		// Accumulate result
    		for (int i = 1; i < frag_C.num_elements; i++) 
      			frag_C.x[i] += frag_C.x[i - 1];
    	}

    	// Store the output
	wmma::store_matrix_sync(C + col_C + row_C * lead_C, frag_C, lead_C, wmma::mem_row_major); 
}

// This kernel converts the input matrices to half precision (supported by wmma)
__global__ void convert_to_fp16(half *OUT, float *__restrict__ IN, const int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N)
		OUT[tid] = IN[tid];
}

// Main function
int main(int argc, char *argv[])
{
	// Set the seed
	srand(21);

	// Input VBS matrix
	VBS vbs;

	// Generate random VBS
	random_sparse_blocks_mat(vbs, M_GLOBAL, K_GLOBAL, 0, 0, WMMA_M, WMMA_K, 0.5f, 0.5f);

	// Print VBS
	fprintf(stdout, "\nInput VBS matrix:\n");
	matprint(vbs);

	// VBS variables
	const int ROW_BLOCKS = vbs.block_rows;
	const int COL_BLOCKS = vbs.block_cols;
	int *VBS_NZCOUNT = vbs.nzcount;
	int *VBS_JAB = vbs.jab;
	int NZ_ELEMENTS = 0;
	int NZ_BLOCKS = 0;

	for (int i = 0; i < ROW_BLOCKS; ++i)
		NZ_BLOCKS += vbs.nzcount[i];

	NZ_ELEMENTS = NZ_BLOCKS * (WMMA_M * WMMA_K);

	fprintf(stdout, "VBS.block_rows = %d\n", ROW_BLOCKS);

	fprintf(stdout, "VBS.block_cols = %d\n", COL_BLOCKS);

	fprintf(stdout, "VBS.nz_blocks = %d\n", NZ_BLOCKS);

	for (int i = 0; i < ROW_BLOCKS; ++i)
		fprintf(stdout, "\nVBS.nzcount[%d] = %d", i, vbs.nzcount[i]);

	fprintf(stdout, "\n");

	fprintf(stdout, "\nVBS.jab:\n");

	for (int i = 0; i < NZ_BLOCKS; ++i)
		fprintf(stdout, "%d ", vbs.jab[i]);

	fprintf(stdout, "\n");

	fprintf(stdout, "\nCompressed VBS.mab:");

	for (int i = 0; i < NZ_ELEMENTS; ++i)
		fprintf(stdout, "%d ", (int)vbs.mab[i]);

	// Host matrices
	float *h_A = NULL;
	float *h_B = NULL;
	float *h_C = NULL;

	// Device matrices
	float *d_Afp32 = NULL;
	half *d_Afp16 = NULL;
	float *d_Bfp32 = NULL;
	half *d_Bfp16 = NULL;
	float *d_Cfp32 = NULL;

	fprintf(stdout, "\nNZ_ELEMENTS = %d\n", NZ_ELEMENTS);

	// Size of matrices
	size_t size_Afp32 = NZ_ELEMENTS * sizeof(float);
	size_t size_Bfp32 = K_GLOBAL * N_GLOBAL * sizeof(float);
	size_t size_Cfp32 = M_GLOBAL * N_GLOBAL * sizeof(float);

	size_t size_Afp16 = NZ_ELEMENTS * sizeof(half);
	size_t size_Bfp16 = K_GLOBAL * N_GLOBAL * sizeof(half);

	// Event timer
	float millis = 0.0f;
	cudaEvent_t start, stop;

	// Create CUDA event
	cudaCheck(cudaEventCreate(&start));
	cudaCheck(cudaEventCreate(&stop));

	// Allocate pinned memory
	cudaCheck(cudaMallocHost(&h_A, size_Afp32));
	cudaCheck(cudaMallocHost(&h_B, size_Bfp32));
	cudaCheck(cudaMallocHost(&h_C, size_Cfp32));

	// Input Compressed VBS values
	h_A = vbs.mab;

	// Input Dense Activation Matrix
	h_B = new float[K_GLOBAL * N_GLOBAL]{0};

	// Fill the random dense activation matrix (all 1's for now)
	random_mat(h_B, K_GLOBAL, N_GLOBAL, 1);

	// Print the input dense activation matrix
	fprintf(stdout, "\n\nInput Dense Activation Matrix:\n");
	matprint(h_B, K_GLOBAL, N_GLOBAL, K_GLOBAL, 1);

	// Check the final result
	// host_multiply(h_A, h_B, h_C, M_GLOBAL, N_GLOBAL, K_GLOBAL);

	// ------------------------------ Allocate Device Memory ------------------------------

	// Device matrix A
	cudaCheck(cudaMalloc(&d_Afp32, size_Afp32));
	cudaCheck(cudaMalloc(&d_Afp16, size_Afp16));

	// Device matrix B
	cudaCheck(cudaMalloc(&d_Bfp32, size_Bfp32));
	cudaCheck(cudaMalloc(&d_Bfp16, size_Bfp16));

	// Copy the host matrices to the device
	cudaCheck(cudaMemcpy(d_Afp32, h_A, size_Afp32, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_Bfp32, h_B, size_Bfp32, cudaMemcpyHostToDevice));

	// Define threads and blocks sizes
	const unsigned int threads_A = M_GLOBAL;
	const unsigned int threads_B = K_GLOBAL;
	const unsigned int blocks_A = K_GLOBAL;
	const unsigned int blocks_B = N_GLOBAL;

	// A Tensor Core allows 2 fp16 input matrices and fp32 accumulation matrices
	convert_to_fp16<<<blocks_A, threads_A>>>(d_Afp16, d_Afp32, NZ_ELEMENTS);
	convert_to_fp16<<<blocks_B, threads_B>>>(d_Bfp16, d_Bfp32, K_GLOBAL * N_GLOBAL);

	// Synchronize host with the device and check for errors
	cudaError_t cudaerr_fp16 = cudaDeviceSynchronize();
	if (cudaerr_fp16 != cudaSuccess)
		fprintf(stdout, "CUDA kernel convert_to_fp16 failed with error: %s.\n", cudaGetErrorString(cudaerr_fp16));

	// --------------- Test for WMMA ---------------

	// Number of blocks in grid (gridDim) and number of threads in block (blockDim)
	dim3 gridDim;
	dim3 blockDim;

	// Here blockDim is used to 
	blockDim.x = ROW_BLOCKS;
	blockDim.y = COL_BLOCKS;

	// NOTE: Here WARP_SIZE is used instead of WARP_SIZE because it would be an undefined reference to __device_builtin_variable_WARP_SIZE
	//gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / WARP_SIZE - 1)) / (WMMA_M * blockDim.x / WARP_SIZE);
	//gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

	// The number of threads depends on the blockDim
	const int THREADS = WARP_SIZE;
	
	// Device matrix C
	cudaCheck(cudaMalloc(&d_Cfp32, size_Cfp32));

	fprintf(stdout, "\nCUDA kernel WMMA_test running ... \n");

	// Start event timer
	cudaCheck(cudaEventRecord(start));

	wmma_test_simple<<< blockDim, THREADS >>>(d_Afp16, d_Bfp16, d_Cfp32, M_GLOBAL, N_GLOBAL, K_GLOBAL, ROW_BLOCKS, COL_BLOCKS, NZ_BLOCKS, VBS_NZCOUNT, VBS_JAB, NZ_ELEMENTS);

	cudaCheck(cudaEventRecord(stop));

	// Error check for WMMA_test kernel
	cudaError_t cudaerr_WMMA = cudaDeviceSynchronize();
	if (cudaerr_WMMA != cudaSuccess)
		fprintf(stdout, "CUDA kernel WMMA_test failed with error: %s.\n", cudaGetErrorString(cudaerr_WMMA));
	else
		fprintf(stdout, "CUDA kernel WMMA_test terminated succesfully!\n");

	// Transfer the output matrix to the host
	cudaCheck(cudaMemcpy(h_C, d_Cfp32, M_GLOBAL * N_GLOBAL * sizeof(float), cudaMemcpyDeviceToHost));

	cudaCheck(cudaEventSynchronize(stop));

	cudaCheck(cudaEventElapsedTime(&millis, start, stop));

	fprintf(stdout, "The WMMA_test function took %f milliseconds.\n", millis);

	// Print the output matrix h_C
	fprintf(stdout, "\n--------------- Printing matrix h_C ---------------\nRows = %d\nCols = %d\n", M_GLOBAL, N_GLOBAL);

	for (int i = 0; i < M_GLOBAL; ++i)
	{
		for (int j = 0; j < N_GLOBAL; ++j)
			fprintf(stdout, "%d ", (int)h_C[i * N_GLOBAL + j]);

		fprintf(stdout, "\n");
	}

	fprintf(stdout, "\nApproximated TFLOPS = %f\n\n", (unsigned long)(2 * M_GLOBAL * N_GLOBAL * K_GLOBAL) / millis / 1e9);

	// For testing size: assert(size == M_GLOBAL * N_GLOBAL);

	// --------------- Free host and device memory ---------------

	// Destroy CUDA event
	cudaCheck(cudaEventDestroy(start));
	cudaCheck(cudaEventDestroy(stop));

	// Free pinned memory
	cudaCheck(cudaFreeHost(h_A));
	cudaCheck(cudaFreeHost(h_B));
	cudaCheck(cudaFreeHost(h_C));

	// Free device memory
	cudaCheck(cudaFree(d_Afp32));
	cudaCheck(cudaFree(d_Afp16));
	cudaCheck(cudaFree(d_Bfp32));
	cudaCheck(cudaFree(d_Bfp16));
	cudaCheck(cudaFree(d_Cfp32));

	// Destroy all allocations and reset device state
	cudaCheck(cudaDeviceReset());

	return 0;
}
