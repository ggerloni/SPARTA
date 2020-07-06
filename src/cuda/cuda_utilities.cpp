// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>


// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#include<stdio.h>
#include <iostream>


#include "cuda_utilities.h"
#include "comp_mats.h"
#include "sparse_utilities.h"

void cublas_blockmat_multiply(const VBS &vbmatA, float *B, int B_cols, int B_lead_dim, float *C, int C_lead_dim){
    //multiplies a VBS matrix (vbmatA) and a dense matrix (B); stores into (C)
    //vbmatA:       column-major entries storage;
    //              column-major block_storage; 
    //B:            column-major storage; TODO: allow general storage format (implement through cublas transpose)
    //C:            column-major storage; TODO: allow general storage format (implement through cublas transpose)

    int A_rows = vbmatA.row_part[vbmatA.block_rows];
    int A_cols = vbmatA.col_part[vbmatA.block_cols];

    int mat_idx = 0; //keeps writing position for mat
    int vbmat_idx = 0; //keeps reading position for vbmat 
    int ja_count = 0; //keeps total nonzero blocks count;

    int B_rows = A_cols;

    int C_rows = A_rows;
    int C_cols = B_cols;

    const float alpha = 1.0f;
    const float beta = 1.0f;
   
    int tot_nonzero_blocks = 0; //index for total nonzero blocks

    int rows_in_block, cols_in_block;
    unsigned int size_block, mem_size_block;

    //TODO: allocate memory on device
    unsigned int size_A = vbmatA.nztot; //total nonzero entries in vbmat
    unsigned int mem_size_A = sizeof(float) * size_A;

    unsigned int size_B = B_rows * B_cols;
    unsigned int mem_size_B = sizeof(float) * size_B;

    unsigned int size_C = C_rows * C_cols;
    unsigned int mem_size_C = sizeof(float) * size_C;

    cublasHandle_t handle;

    checkCudaErrors(cublasCreate(&handle));

    float* d_A, * d_B, * d_C;
    checkCudaErrors(cudaMalloc((void**)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void**)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void**)&d_C, mem_size_C));

    //copy to device the vbmat matrix (nonzero blocks are stored consecutively and in column major format)
    checkCudaErrors(cublasSetVector(
        size_A, sizeof(float), vbmatA.mab, 1, d_A, 1));

    //copy B to device (maybe more efficient to copy it block by block?)
    checkCudaErrors(cublasSetMatrix(
        B_rows, B_cols, sizeof(float), B, B_lead_dim, d_B, B_rows));


    //creates streams. Each block rows is assigned a different stream.
    cudaStream_t streams[vbmatA.block_rows];
    for (int ib = 0; ib < vbmatA.block_rows; ib++)
    {
        cudaStreamCreate(&(streams[ib]));
    }

    //loop through all blocks
    for(int jb = 0; jb < vbmatA.block_cols; jb++ )      //loop horizontally through block columns
    {
        cols_in_block = vbmatA.col_part[jb+1] - vbmatA.col_part[jb];
        const float* d_B_block = d_B + vbmatA.col_part[jb];    //access the block of B that is going to be multiplied with blocks of A in column jb

        for(int nzs = 0; nzs < vbmatA.nzcount[jb]; nzs++)        //loop vertically through nonzero blocks

        {

            int ib = vbmatA.jab[tot_nonzero_blocks];             //the block row position of a nonzero block 
            tot_nonzero_blocks += 1;
            rows_in_block = vbmatA.row_part[ib + 1] - vbmatA.row_part[ib]; //the row height of the block

            cublasSetStream(handle, streams[ib]);               //each block row works on a different stream

            //define the sub-matrices
	        const float* d_A_block = d_A + vbmat_idx;           //access the block on d_A.
            float* d_C_block = d_C + vbmatA.row_part[ib] ;  //access the block on d_C.

            //multiply the blocks, store result in d_C_block
            checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                rows_in_block, B_cols, cols_in_block,           //m, n, k <-- block_A: m*k   block_B: k*n   block_C: m*n
                &alpha,
                d_A_block, rows_in_block,                        //blockA device pointer, leading dimension
                d_B_block, B_rows,                              //blockB device pointer, leading dimension
                &beta,
                d_C_block, C_rows));                            //blockC device pointer, leading dimension

            vbmat_idx += rows_in_block * cols_in_block;
	    }

    }

    //let each stream copy the relevant C block from device
    for (int ib = 0; ib < vbmatA.block_rows; ib++)
    {
        cublasSetStream(handle, streams[ib]); 
        rows_in_block = vbmatA.row_part[ib + 1] - vbmatA.row_part[ib];
        checkCudaErrors(cublasGetMatrix(
            rows_in_block, C_cols, sizeof(float), d_C + vbmatA.row_part[ib], C_rows, C + vbmatA.row_part[ib], C_lead_dim));

    }

    cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));

    checkCudaErrors(cublasDestroy(handle));

}

//Matrix-Matrix multiplication with cublas. A,B,C are in column-major order.
//Matrix A and B are in host
//Matrix d_C is in device to allow for accumulation of results
int cublas_gemm_custom(const float *A, unsigned int A_rows, unsigned int A_cols, unsigned int lda,
	const float *B, unsigned int B_cols, unsigned int ldb,
	float *C, unsigned int ldc,
	const float alpha,
	const float beta)
{
    
    //deduce matrices dimensions
    unsigned int B_rows = A_cols;
    unsigned int C_rows = A_rows;
    unsigned int C_cols = B_cols;


    //allocate memory on device
    //-------------------------------------------------------
    unsigned int size_A = A_rows * A_cols;
    unsigned int mem_size_A = sizeof(float) * size_A;

    unsigned int size_B = B_rows * B_cols;
    unsigned int mem_size_B = sizeof(float) * size_B;    
  
    unsigned int size_C = C_rows * C_cols;
    unsigned int mem_size_C = sizeof(float) * size_C;

    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B)); 
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));
    //-------------------------------------------------------

    //copy matrices to device
    checkCudaErrors(cublasSetMatrix(
                                    A_rows, A_cols, sizeof(float), A, lda, d_A, A_rows));
    checkCudaErrors(cublasSetMatrix(
                                    B_rows, B_cols, sizeof(float), B, ldb, d_B, B_rows));

    // CUBLAS version 2.0
    cublasHandle_t handle;

    checkCudaErrors(cublasCreate(&handle));

    //Perform gemm operation with cublas
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                A_rows, B_cols, A_cols,
                                &alpha,
                                d_A, A_rows,
                                d_B, B_rows,
                                &beta,
                                d_C, C_rows));

    // copy result from device to host 
    checkCudaErrors(cublasGetMatrix(C_rows, C_cols, sizeof(float), d_C, C_rows, C, C_rows));

    cudaDeviceSynchronize();

    // clean up memory
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));

    // Destroy the handle
    checkCudaErrors(cublasDestroy(handle));
	
    return 0;
}


int cusparse_gemm_custom(const CSR& cmat, float* B, int B_cols, int B_lead_dim, float* C, int C_lead_dim, 
    const float alpha,
    const float beta)
{
    if (cmat.fmt != 0)
    {
        std::cout << "ERROR: cusparse_gemm_custom only supports CSR (row-major) " << std::endl;
        return 1;
    }


    //fill csrRowPtr (element i holds number of nonzero entries up to row i)
    int csrRowPtr[cmat.rows + 1];
    csrRowPtr[0] = 0;

    int nnz = 0;
    for (int i = 0; i < cmat.rows; i++)
    {
        nnz += cmat.nzcount[i];
        csrRowPtr[i + 1] = nnz;
        std::cout << csrRowPtr[i + 1] << std::endl;
   
    }
    //-------------------------------------------------------------

    //fill csrVal (the nonzero values) and csrColInd (their column indices)
    int csrColInd[nnz];
    float csrVal[nnz];
    nnz = 0;
    for (int i = 0; i < cmat.rows; i++)
    {
        nnz += cmat.nzcount[i];
        std::copy(cmat.ja[i], cmat.ja[i] + cmat.nzcount[i], csrColInd + nnz);
        std::copy(cmat.ma[i], cmat.ma[i] + cmat.nzcount[i], csrVal + nnz);
    }

    matprint(cmat);

    //allocate memory on device
    unsigned int mem_size_csrVal = sizeof(float) * nnz;
    unsigned int mem_size_csrColInd = sizeof(int) * nnz;
    unsigned int mem_size_csrRowPtr = sizeof(int) * (cmat.rows + 1);

    unsigned int B_rows = cmat.cols;
    unsigned int size_B = B_rows * B_cols;
    unsigned int mem_size_B = sizeof(float) * size_B;
    
    unsigned int C_rows = cmat.rows;
    unsigned int C_cols = B_cols;
    unsigned int size_C = C_rows * C_cols;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate device memory
    int* d_RowPtr, * d_ColInd;
    float* d_Val;

    checkCudaErrors(cudaMalloc((void**)&d_RowPtr, mem_size_csrRowPtr));
    checkCudaErrors(cudaMalloc((void**)&d_ColInd, mem_size_csrColInd));
    checkCudaErrors(cudaMalloc((void**)&d_Val, mem_size_csrVal));

    float * d_B, * d_C;
    checkCudaErrors(cudaMalloc((void**)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void**)&d_C, mem_size_C));

    //copy arrays and matrices to device
    checkCudaErrors(cublasSetVector(
        nnz, sizeof(float),
        csrVal, 1, d_Val, 1));

    for (int i = 0; i < nnz; i++)
    {
        std::cout << csrVal[i] << " ";
    }
    std::cout << std::endl;

    checkCudaErrors(cublasSetVector(
        nnz, sizeof(int),
        csrColInd, 1, d_ColInd, 1));

    arr_print(csrColInd, nnz);


    checkCudaErrors(cublasSetVector(
        (cmat.rows + 1), sizeof(int),
        csrRowPtr, 1, d_RowPtr, 1));

    arr_print(csrRowPtr, cmat.rows + 1);



    checkCudaErrors(cublasSetMatrix(
        B_rows, B_cols, sizeof(float), B, B_lead_dim, d_B, B_rows));


    if (beta != 0)
    {
        checkCudaErrors(cublasSetMatrix(
            C_rows, C_cols, sizeof(float), C, C_lead_dim, d_C, C_rows));
    }

    cusparseHandle_t handle;
    cusparseMatDescr_t descrA;

    checkCudaErrors(cusparseCreateMatDescr(&descrA));
    checkCudaErrors(cusparseCreate(&handle));
    
    checkCudaErrors(
        cusparseScsrmm(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            cmat.rows,
            B_cols,
            cmat.cols,
            csrRowPtr[cmat.rows + 1],
            &alpha,
            descrA,
            d_Val,
            d_RowPtr,
            d_ColInd,
            d_B,
            B_rows,
            &beta,
            d_C,
            C_rows)
    );


    cudaDeviceSynchronize();

    // copy result from device to host 
    checkCudaErrors(cublasGetMatrix(C_rows, C_cols, sizeof(float), d_C, C_rows, C, C_rows));

    // clean up memor;y
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_Val));
    checkCudaErrors(cudaFree(d_RowPtr));
    checkCudaErrors(cudaFree(d_ColInd));

    // Destroy the handle
    checkCudaErrors(cusparseDestroy(handle));

}