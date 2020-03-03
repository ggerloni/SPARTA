# SPARTA
**SP**arse **A**ccele**R**ation on **T**ensor **A**rchitecture

The repository contains code for reordering sparse matrices so that they contain dense blocks. This allow to manipulate (e.g. multiplicate) these matrices faster on dense-specific architecture. 

Sparse matrices are efficiently stored in SparMat objects, which follow the Compressed Sparse Row (CSR) format. block-sparse matrices are strored in VBSparMat object, following the Variable Block format. Several tools are provided for input/output and conversion among Sparmat, VBSparMat and several other well known matrix formats such as Snap and Matrix Market.

The code requires Intel MKL to run.  Remember to configure the link command through
[Intel MKL Link Line Advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/)

running `make` will create a test executable (from Test.cpp) in the programs/ folder. You can run it with different command line arguments to test different features.  



Source files are stored in the src folder:
* Test.cpp shows an example of how reading, conversion, rows reorder and sparse-dense multiplication work for several input matrix format.
* setblks.cpp contains tools to handle interactions (conversion, reorder) between SparMat and VBSparMat objects.
* utilities.cpp contains other tools (mainly interaction with SparMat objects). 

