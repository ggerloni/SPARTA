#include <stdio.h>
#include <mma.h>

// Custom error checking macro
#define cudaCheck(stat)                         \
	{                                           \
		cudaCheck_((stat), __FILE__, __LINE__); \
	}
void cudaCheck_(cudaError_t stat, const char *file, int line)
{
	if (stat != cudaSuccess)
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
}

// The warpSize cannot be called inside host functions -> undefined reference to __device_builtin_variable_warpSize
#define WARP_SIZE 32

// The amount of matrix data of A and B to be cached in shared memory
// The value is divided in half because of skew overhead
#define CHUNK_K 8

// This value is used for shifting rows of matrix A and cols of matrix B to reduce possible bank conflicts
#define SKEW 16

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

// Define the context of functions and types for the nvcuda::wmma namespace
using namespace nvcuda;

// NOTE: Further optimization -> cudaCheck(cudaFuncSetAttribute(WMMA_test_opt, cudaFuncAttributeMaxDynamicSharedMemorySize, 64));

// TODO: - Avoid random global memory access when the CTA copies the tiles of matrix C
//       - Copy portions of matrices A and B on each iteration from global memory to shared memory using padding, i.e. skew, to prevent shared memory bank conflicts
//       - When CTA is finished, copy the results in shared memory back to the global memory

// An optimized version of the WMMA_test kernel
__global__ void WMMA_test_opt(const half *__restrict__ A, const half *__restrict__ B, const float *C, const int M, const int K, const int N)
{
	// Shared memory: cache data for matrices A and B
	extern __shared__ half S[][CHUNK_K * MATRIX_K + SKEW];

	// Leading dimensions
	int lead_A = M;
	int lead_B = K;
	int lead_C = M;

	/ Offset in shared memory from which the B matrix is stored.const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

	// This pointer is used to access the C and D matrix tiles this warp computes.
	float *shmem_warp_tile_ptr = (float *)&shmem[0][0] + (warpId / 2) * SHMEM_STRIDE * K * 2 + (warpId % 2) * SHMEM_OFFSET;

	// This pointer is used to stream the C and D matrices block-wide tile to and
	// from shared memory.
	float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

	// Adjust the beta scaler, as it'll be multiplied by alpha at the end of
	// each tile computation. Technically this is not generally correct (may
	// result in a loss of precision). Zero still needs to be specially handled
	// though.
	beta /= alpha;

	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.
	for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x)
	{
		const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

		// Stop when there are no more D matrix tiles to compute in this CTA.
		if (block_tile_i >= M_TILES)
			break;

		// This warp's pointer to the C matrix data to copy memory from to shared
		// memory.
		const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
		const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

		// Stream multiple C tiles to shared memory.
#pragma unroll
		for (int i = 0; i < K; i++)
		{
			typedef int4 copy_t;

			*((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
				*((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
				  laneId);
		}

		__syncthreads();

		// Accumulate the result of A and B matrix fragment
		wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES] [WARP_ROW_TILES];

		// Load the C matrix tiles into fragments from shared memory.
#pragma unroll
		for (int i = 0; i < WARP_COL_TILES; i++)
		{
#pragma unroll
			for (int j = 0; j < WARP_ROW_TILES; j++)
			{
				const float *tile_ptr =
					shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

				wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
			}
		}

		__syncthreads();

		// Scale the C matrix.
#pragma unroll
		for (int i = 0; i < WARP_COL_TILES; i++)
		{
#pragma unroll
			for (int j = 0; j < WARP_ROW_TILES; j++)
			{
#pragma unroll
				for (int t = 0; t < c[i][j].num_elements; t++)
				{
					c[i][j].x[t] *= beta;
				}
			}
		}

		// Select what warp copies what matrix to shared memory.
		// Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
		const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
											   M * K_GLOBAL * (warpId % 4) * 2)
											: (&B[block_tile_j * N * K_GLOBAL] +
											   N * K_GLOBAL * (warpId % 4) * 2);

		// Go through the global K dimension by a fixed step at a time.
#pragma unroll
		for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K)
		{
			// Copy slices of the A and B matrices to shared memory.
			// The first half of the warps in the CTA copy the A matrix, the rest copy
			// the B matrix.
			size_t shmem_idx =
				warpId < (WARPS_PER_BLOCK / 2)
					? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
					: (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

			// First half of the warp copies the first row / column of the matrix,
			// the second half of the warp copies the next.
			int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
									  (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
							 (laneId % CHUNK_COPY_LINE_LANES);

			// Shift the second half of the warp to the next row / column in the
			// shared memory.
			shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
			for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
				 i++)
			{
				// Copy 16 bytes at once in each lane.
				*((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
					*lane_ptr;

				// Advance the global memory pointer and the shared memory index.
				lane_ptr =
					(int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
				shmem_idx += CHUNK_COPY_LINES_PER_WARP;
			}

			__syncthreads();

			// Compute a grid of C matrix tiles in each warp.
#pragma unroll
			for (int k_step = 0; k_step < CHUNK_K; k_step++)
			{
				wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major>
					a[WARP_COL_TILES];
				wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major>
					b[WARP_ROW_TILES];

#pragma unroll
				for (int i = 0; i < WARP_COL_TILES; i++)
				{
					size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
					const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

					wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
					for (int j = 0; j < WARP_ROW_TILES; j++)
					{
						if (i == 0)
						{
							// Load the B matrix fragment once, because it is going to be
							// reused against the other A matrix fragments.
							size_t shmem_idx_b = shmem_idx_b_off +
												 (WARP_ROW_TILES * N) * (warpId % 2) +
												 (j * N);
							const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

							wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
						}

						wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
					}
				}
			}

			__syncthreads();
		}

		// Store the D fragments to shared memory.
#pragma unroll
		for (int i = 0; i < WARP_COL_TILES; i++)
		{
#pragma unroll
			for (int j = 0; j < WARP_ROW_TILES; j++)
			{
#pragma unroll
				// Uniform, point-wise transformations of ALL fragment elements by ALL
				// threads in the warp are well-defined even though element indices
				// within fragment storage are not defined.
				for (int t = 0; t < c[i][j].num_elements; t++)
					c[i][j].x[t] *= alpha;

				float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

				wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
			}
		}

		__syncthreads();

		// Now that shared memory contains all the D tiles, stream them to global
		// memory.
		float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
		for (int i = 0; i < K; i++)
		{
			*((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
				*((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
		}

		__syncthreads();
	}
}

// Convertion from fp32 (float precision) to fp16 (half precision)
__global__ void convertFp32ToFp16(half *OUT, float *IN, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N)
		OUT[tid] = IN[tid];
}

// Generate a dense matrix with values X
void fillRandom(float *M, int ROW, int COL, float X)
{
	for (int i = 0; i < ROW; ++i)
		for (int j = 0; j < COL; ++j)
			M[i * COL + j] = X;
}

// Main function
int main(int argc, char *argv[])
{
	// Host matrices -> 2 inputs, 1 output
	float *h_A;
	float *h_B;
	float *h_C;

	// Device matrices -> 2 fp32, 2 fp16 and 1 fp32 (output)
	float *d_Afp32;
	half *d_Afp16;
	float *d_Bfp32;
	half *d_Bfp16;
	float *d_Cfp32;

	// Size of matrices
	size_t size_Afp32 = MATRIX_M * MATRIX_K * sizeof(float), size_Afp16 = size_Afp32 / 2;
	size_t size_Bfp32 = MATRIX_K * MATRIX_N * sizeof(float), size_Bfp16 = size_Bfp32 / 2;
	size_t size_Cfp32 = MATRIX_M * MATRIX_N * sizeof(float);

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

	// NOTE: allocation without pinned memory -> h_A = new float[MATRIX_M * MATRIX_N] { 0 };
	// Generate 2 random fp32 input matrices
	fillRandom(h_A, MATRIX_M, MATRIX_K, 1.0);
	fillRandom(h_B, MATRIX_K, MATRIX_N, 1.0);

	//--------------- Allocate input matrices ---------------

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
	const int threads_A = 128;
	const int threads_B = 128;
	const int blocks_A = (MATRIX_M * MATRIX_K) / threads_A;
	const int blocks_B = (MATRIX_K * MATRIX_N) / threads_B;

	// A Tensor Core allows 2 fp16 input matrices and fp32 accumulation matrices
	convertFp32ToFp16<<<blocks_A, threads_B>>>(d_Afp16, d_Afp32, MATRIX_M * MATRIX_K);
	convertFp32ToFp16<<<blocks_B, threads_B>>>(d_Bfp16, d_Bfp32, MATRIX_K * MATRIX_N);

	// Synchronize host with the device and check for errors
	cudaError_t cudaerr_fp16 = cudaDeviceSynchronize();
	if (cudaerr_fp16 != cudaSuccess)
		fprintf(stdout, "CUDA kernel convertFp32ToFp16 failed with error: %s.\n", cudaGetErrorString(cudaerr_fp16));

	// --------------- Test for WMMA ---------------

	// Number of blocks in grid (gridDim) and number of threads in block (blockDim)
	dim3 gridDim;
	dim3 blockDim;

	// NOTE: The blockDim.x must be a multiple of WARP_SIZE
	// In this case, 128 * 4 -> we have 16 warps and 1 block computes a 64 * 64 tile
	blockDim.x = 128;
	blockDim.y = 4;

	// NOTE: Here WARP_SIZE is used instead of warpSize because it would be an undefined reference to __device_builtin_variable_warpSize
	gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / WARP_SIZE - 1)) / (WMMA_M * blockDim.x / WARP_SIZE);
	gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

	// Device matrix C
	cudaCheck(cudaMalloc(&d_Cfp32, size_Cfp32));

	fprintf(stdout, "CUDA kernel WMMA_test_opt running ... \n");

	// Start event timer
	cudaCheck(cudaEventRecord(start));

	WMMA_test_opt<<<gridDim, blockDim>>>(d_Afp16, d_Bfp16, d_Cfp32, MATRIX_M, MATRIX_K, MATRIX_N);

	cudaCheck(cudaEventRecord(stop));

	// Error check for WMMA_test_opt kernel
	cudaError_t cudaerr_WMMA = cudaDeviceSynchronize();
	if (cudaerr_WMMA != cudaSuccess)
		fprintf(stdout, "CUDA kernel WMMA_test_opt failed with error: %s.\n", cudaGetErrorString(cudaerr_WMMA));
	else
		fprintf(stdout, "CUDA kernel WMMA_test_opt terminated succesfully!\n");

	// Transfer the output matrix to the host
	cudaCheck(cudaMemcpy(h_C, d_Cfp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

	cudaCheck(cudaEventSynchronize(stop));

	cudaCheck(cudaEventElapsedTime(&millis, start, stop));

	fprintf(stdout, "The WMMA_test_opt function took %f milliseconds.\n", millis);

	// Print the output matrix h_C
	for (int i = 0; i < MATRIX_M; ++i)
	{
		for (int j = 0; j < MATRIX_N; ++j)
			fprintf(stdout, "%d ", (int)h_C[i * MATRIX_N + j]);

		fprintf(stdout, "\n");
	}

	// Size test: assert(size == MATRIX_M * MATRIX_N);

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
