// Parallelization of the matrix addition program using the gpu and cuda.

#include<cuda.h> // include the cuda environment.
#include<stdio.h>
#include<stdlib.h>

// set size to 10.
const int SIZE = 10;
const int BLOCKS = 1; // only launch kernels on one block.
const dim3 threadsPerBlock(SIZE, SIZE); // launch kernels on threads of
                                        // dimensions SIZE x SIZE

// Populate the matrices.
__global__ void mat_populate(int mat1[][SIZE], int mat2[][SIZE], int mat3[][SIZE]){

    int i = threadIdx.x;
    int j = threadIdx.y;

    // Ensure thread ids don't exceed the given size parameter.
    if((i < SIZE) && (j < SIZE)){
            mat1[i][j] = i+j; // add i and j to be entry ij.
            mat2[i][j] = i*j; // multiply i and j to be entry ij
            mat3[i][j] = 0; // populate all entries with 0.
    }

    return;
}

// add mat1 and mat2 and store in mat3.
__global__ void mat_add(int mat1[][SIZE], int mat2[][SIZE], int mat3[][SIZE]){

    int i = threadIdx.x;
    int j = threadIdx.y;

    // Ensure thread ids don't exceed the given size parameter.
    if((i < SIZE) && (j < SIZE)){
        mat3[i][j] = mat1[i][j]+mat2[i][j];
    }

    return;
}

// assert whether addition was perfomed correctly.
// if assert stores 0, success, else failure at entry ij.
__global__ void mat_add_assert(int mat1[][SIZE], int mat2[][SIZE], int mat3[][SIZE], int *assert){

    *assert = 0;

    int i = threadIdx.x;
    int j = threadIdx.y;

    // Ensure thread ids don't exceed the given size parameter.
    if((i < SIZE) && (j < SIZE)){
        if(!(mat3[i][j] == mat1[i][j]+mat2[i][j])){
            *assert = 1;

            return;
        }
    }

    return;
}

// Print the entries of the matrices in matrix format.
void mat_print(int mat[][SIZE]){

    for(int i = 0; i < SIZE ; i++){
        for(int j = 0; j < SIZE ; j++){
            printf("%d\t", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    return;
}

int main(){

    // declare and initialize assertion variable and accompying device pointer.
    int assert;
    int *d_assert = 0;

    // allocate size of assert to device pointer on the gpu.
    // copy the data on assert to device pointer d_assert.
    cudaMalloc(&d_assert, sizeof(assert));
    cudaMemcpy(d_assert, &assert, sizeof(assert), cudaMemcpyHostToDevice);

    // declare matrices of size SIZE x SIZE
    int mat1[SIZE][SIZE];
    int mat2[SIZE][SIZE];
    int mat3[SIZE][SIZE];

    // declare and initialize associated device pointers.
    int (*d_mat1)[SIZE] = 0;
    int (*d_mat2)[SIZE] = 0;
    int (*d_mat3)[SIZE] = 0;

    // allocate SIZE*SIZE bytes to device pointers on gpu.
    cudaMalloc(&d_mat1, sizeof(mat1));
    cudaMalloc(&d_mat2, sizeof(mat2));
    cudaMalloc(&d_mat3, sizeof(mat3));

    // copy contents of mat1, mat2, and mat3 to device pointers, respectively.
    cudaMemcpy(d_mat1, &mat1, sizeof(mat1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, &mat2, sizeof(mat2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat3, &mat3, sizeof(mat3), cudaMemcpyHostToDevice);

    // populate the matrices on the gpu.
    // add the matrices d_mat1 and d_mat2, store in d_mat3.
    // Assert that addition was successful, store in d_assert.
    mat_populate<<<BLOCKS, threadsPerBlock>>>(d_mat1, d_mat2, d_mat3);
    mat_add<<<BLOCKS, threadsPerBlock>>>(d_mat1, d_mat2, d_mat3);
    mat_add_assert<<<BLOCKS, threadsPerBlock>>>(d_mat1, d_mat2, d_mat3, d_assert);

    // Copy assertion data in d_assert to assert.
    cudaMemcpy(&assert, d_assert, sizeof(assert), cudaMemcpyDeviceToHost);

    // if assert=0, print success, else, print failure.
    if(!(assert)){
        printf("Assertions passed.\n");
    }
    else{
        printf("Assertions failed.\n");
    }

    // free all device pointers.
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_mat3);
    cudaFree(d_assert);

    return 0;
}
