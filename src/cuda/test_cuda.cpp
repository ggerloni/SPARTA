#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <string>
#include <unistd.h>
#include <math.h>
#include <typeinfo>

#include "globheads.h"
#include "protos.h"
#include "utilities.h"

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#include "cuda_utilities.h"


int main(int argc, char *argv[]) {

 
    if (typeid(DataT) != typeid(float)){
        cout<< "WARNING: only float supported for CUDA. Change DataT to float in protos.h" <<endl;
	return 1;
    }
    opterr = 0;

    int input_type = 1;
    int n = 20;             //rows in the square input matrix;
    int out_columns = 5;    //number of columns in the output matrix;
    float sparsity = 0.5;   //sparsity of the input matrix;
    string input_source;
    float eps = 0.5;        //this value sets how different two rows in the same block can be.
                            //eps = 1 means only rows with equal structure are merged into a block
                            //eps = 0 means all rows are merged into a single block
   

    //terminal options loop
    opterr = 0;
    char c;
    while ((c = getopt (argc, argv, "i:s:k:o:n:e:")) != -1)
      switch (c)
        {
        case 'i':// select input example
            input_type = stoi(optarg);
            //  1: Random CSR
            //  2: SNAP Edgelist
            //  3: MTX Format
            //  4: Random Variable Block matrix
            if(input_type < 1 or input_type > 4){
                input_type = 0;
                cout<<"WARNING: invalid input reference. Using 1 (Random CSR)"<<endl;
            }
            break;
        
        case 's': //select source file
            //has only effect for example 2 and 3;
            input_source = optarg;
            break;
        
        case 'k': //input matrix sparsity
            //has only effect for example 1 and 4
            sparsity = stof(optarg);
                if(sparsity < 0 or sparsity > 1){
                    fprintf (stderr, "Option -k tried to set sparsity outside of [0,1]");
                    return 1;
                }
          break;
                
        case 'n': //input matrix dimension
             //has only effect for example 1 and 4
            n = stoi(optarg);
	    break;
        
        case 'o': //number of column of output matrix
            out_columns = stoi(optarg);
            break;

        case 'e': //epsilon used for matrix reordering;
            eps = stof(optarg);
            if(eps < 0. or eps > 1.){
                fprintf (stderr, "Option -e tried to set epsilon outside of [0,1]");
                return 1;
            }
	    break;
                
        case '?':
            fprintf (stderr, "Option -%c does not exists, or requires an argument.\n", optopt);
            return 1;
        default:
          abort ();
	}
    

//INPUT CONVERSION TO Compressed Sparse Row (CSR)

	SparMat spmat; //this will hold the CSR matrix


//INPUT EXAMPLE 1: RANDOM CSR
//create a random sparse matrix
    if (input_type == 1){
        Mat rand_mat;
        random_sparse_mat(rand_mat, n, (float) sparsity); //generate random Mat
        convert_to_CSR(rand_mat, spmat);
        cout << "CREATED A RANDOM CSR" << endl;

    }
//______________________________________

//TODO add import error notification
//INPUT EXAMPLE 2: read graph in edgelist format into CSR
    if (input_type == 2){
        if (input_source.empty()) input_source = "testgraph.txt";
       
        read_snap_format(spmat, input_source);         //Read a CSR matrix from a .txt edgelist (snap format) 
        cout << "IMPORTED A CSR FROM A SNAP EDGELIST" << endl;


    }
 //______________________________________
        
//TODO add import error notification
//INPUT EXAMPLE 3: read from MTX format
    if (input_type == 3){
        //read from mtx
        if (input_source.empty()) input_source = "testmat.mtx";
        read_mtx_format(spmat, input_source); //read into CSR
        
        cout << "IMPORTED A CSR FROM MTX FILE" << endl;
        }


//______________________________________
//INPUT EXAMPLE 4: create a random matrix with block structure
    if (input_type == 4){
    
    //n and sparsity have been previously set by options. Default: n = 20, sparsity = 0.5;
	int n_block = 3; //number of blocks
	float k_block = sqrt(sparsity); //percentage of non-zero blocks,must always be greater than sparsity

	Mat rnd_bmat;
	random_sparse_blocks_mat(rnd_bmat, n, n_block, k_block, sparsity); //generate the random sparse matrix with block structure
    
	convert_to_CSR(rnd_bmat, spmat); //convert it to CSR
        
    cout << "CREATED A RANDOM BLOCK MATRIX" << endl;

	//TODO optional: scramble the matrix row to see if the algo can reorder them.

    }
        
//___________________________________________
//*******************************************
//		END OF INPUT
//spmat must hold a proper CSR matrix at this point
//******************************************

//reorder the CSR matrix spmat and generate a Block Sparse Matrix

    VBSparMat vbmat; //Variable Blocks matrix format
    make_sparse_blocks(spmat, vbmat,eps); //reorders the csr so that it has dense blocks. 
                                          //Then convert the reordered csr into a vbmat
                                          //that has sparse blocks.

    cout<<"CSR permuted. VBSparMat created"<<endl;

//*******************************************
//        REPORT ON BLOCK STRUCTURE
//******************************************
    ofstream CSV_out;
    CSV_out.open("output.txt");

    string CSV_header = "MatrixSize,OriginalSparsity,Divisions,NonzeroBlocks,AvgBlockHeight,AvgBHError,AvgBlockLength,AvgBLError,NonzeroAreaFraction,AverageBlockPopulation,ABPError,NewSparsity"; 
    CSV_out << CSV_header << endl;

    bool verbose = true; //print mat analysis on screen too?
    features_to_CSV(&vbmat, CSV_out, verbose);//write mat analysis on csv
    CSV_out.close();
	

//*******************************************
//         MULTIPLICATION PHASE
//___________________________________________
//several ways of multiplying the sparse matrix
//with a dense one, with benchmarks
//******************************************
    
//create a dense array matrix from spmat (for CUBLAS GEMM)
    	Mat mat;
    	int mat_n = spmat.n;
    	DataT* mat_arr;
	    mat_arr = new DataT[mat_n*mat_n];
    	convert_from_CSR(spmat, mat);
    	std::copy((mat.vec).begin(), (mat.vec).end(), mat_arr);

	    DataT* mat_arr_c = new DataT[mat_n*mat_n]; //column major version of mat_arr
	    convert_to_col_major(mat_arr, mat_arr_c, mat_n, mat_n);

    	cout << fixed; //output format


//TODO make all matrices in column-major format from the start, or better give the choice. 
	    cout << "\n \n **************************** \n STARTING THE MULTIPLICATION PHASE \n" << endl; 

        
        //creating the dense matrix X
	    int X_rows = spmat.n;
	    int X_cols = out_columns;

	    int seed = 123;
  	    srand(seed);
	    DataT X[X_rows*X_cols];
  	    for (int k=0; k<X_rows*X_cols; k++) {
            DataT x =  rand()%100;
    		X[k] = x/100;
  	    }

   	    DataT X_c[X_rows*X_cols]; //column major version of X
    	convert_to_col_major(X, X_c, X_rows, X_cols);


//----------------------------
//creating the output matrix Y
	int Y_rows = spmat.n;
	int Y_cols = X_cols;


//result matrices, to be filled in column-major format

    	DataT Y_gemm[Y_rows * Y_cols];
	//DataT Y_csr[Y_rows * Y_cols];
    	DataT Y_block[Y_rows * Y_cols] = {};
	//DataT Y_batch[Y_rows * Y_cols] = {};

//dense-dense cublas gemm multiplication
    
    	clock_t start_t = clock();

    	unsigned int size_Y = Y_rows * Y_cols;
    	unsigned int mem_size_Y = size_Y * sizeof(float);
  	DataT *d_Y;
   	checkCudaErrors(cudaMalloc((void **) &d_Y, mem_size_Y)); 

    	cublas_gemm_custom (mat_arr_c, mat_n, mat_n, mat_n, X, X_cols, X_rows, d_Y, mat_n);

    	checkCudaErrors(cublasGetMatrix(Y_rows, Y_cols, sizeof(float), d_Y, mat_n, Y_gemm, mat_n));

    	checkCudaErrors(cudaFree(d_Y));
 
    	double total_t = (clock() - start_t)/(double) CLOCKS_PER_SEC;
    	cout<<"Dense-Dense multiplication. Time taken: " << total_t<<endl;


	matprint(mat_arr_c,mat_n,mat_n);

	matprint(vbmat);
//vbr-dense cublas multiplication	
	DataT Y_block_c[X_rows*X_cols] = {};

        start_t = clock();

	cublas_blockmat_multiply(vbmat, X_c, X_cols, Y_block);
	
	total_t = (clock() - start_t)/(double) CLOCKS_PER_SEC;
	
	cout <<"BlockSparse-Dense multiplication. Time taken: " << total_t<<endl;


//TODO CSR-dense cusparse multiplication

 
 
//PRINT RESULTING MATRICES

//	cout << "CSR RESULT" << endl;
//        matprint(&Y_csr[0],spmat.n, X_cols);

	cout << "GEMM RESULT" << endl;
	matprint(&Y_gemm[0],Y_rows, Y_cols);
	
	cout << "BLOCK RESULT" << endl;
	matprint(&Y_block[0],Y_rows, Y_cols);

//	cout << "BLOCK BATCH RESULT" << endl;
//        matprint(&Y_batch[0],spmat.n, X_cols);



}