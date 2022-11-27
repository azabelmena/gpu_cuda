// Parallelization of the vector (array) addition program using cuda.

#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>

// set size to 2**19=524288
const int SIZE = (1 << 19);
const int BLOCKS = 512; // use launch kernel with only 512 blocks.
const int threadsPerBlock(1024); // use 1024 threads per block

// populate the vectors.
__global__ void vec_populate(int *vec1, int *vec2, int *vec3){

    int i = threadIdx.x;

    if(i < SIZE){
        vec1[i] = i*i; // square the index.
        vec2[i] = i*i*i; // cube the index.
        vec3[i] = 0; // populate all indices with 0.
    }

    return;
}

// add vec1 and vec2, store in vec3.
__global__ void vec_add(int *vec1, int *vec2, int *vec3){

    int i = threadIdx.x;

    if(i < SIZE){
        vec3[i] = vec1[i]+vec2[i];
    }

    return;
}

// assert that addition was correct.
// store 0, success, else, failure.
// vlaue stored in assert.
__global__ void vec_add_assert(int *vec1, int *vec2, int *vec3, int *assert){

    *assert = 0;

    int i = threadIdx.x;

    if(i < SIZE){
        if(!(vec3[i] == vec1[i]+vec2[i])){
            *assert = 1;

            return;
        }
    }

    return;
}

// print the contents of the vector.
void vec_print(int *vec){

    for(int i = 0; i < SIZE ; i++){
        printf("%d\t", vec[i]);
    }
    printf("\n");

    return;
}

int main(){
    // set up asseriton variable and initialize accompying device pointer.
    int assert;
    int *d_assert = 0;

    // alllocate space on the gpu for d_assert.
    // copy contents of assert to d_assert.
    cudaMalloc(&d_assert, sizeof(assert));
    cudaMemcpy(d_assert, &assert, sizeof(assert), cudaMemcpyHostToDevice);

    // declare vectors.
    int vec1[SIZE];
    int vec2[SIZE];
    int vec3[SIZE];

    // declare associated device pointers for vec1, vec2, vec3
    int *d_vec1 = 0;
    int *d_vec2 = 0;
    int *d_vec3 = 0;

    // allocate memory on the gpu for device pointers vec1, vec2, vec3.
    cudaMalloc(&d_vec1, sizeof(vec1));
    cudaMalloc(&d_vec2, sizeof(vec2));
    cudaMalloc(&d_vec3, sizeof(vec3));

    // copy contents of vec1, vec2, vec3 to their device pointers.
    cudaMemcpy(d_vec1, &vec1, sizeof(vec1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, &vec2, sizeof(vec2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec3, &vec3, sizeof(vec3), cudaMemcpyHostToDevice);

    // populate d_vec1, d_vec2, d_vec3.
    // add d_vec1 and d_vec2, store in d_vec3.
    // asser that addition was successful, store in d_asssert.
    vec_populate<<<BLOCKS, threadsPerBlock>>>(d_vec1, d_vec2, d_vec3);
    vec_add<<<BLOCKS, threadsPerBlock>>>(d_vec1, d_vec2, d_vec3);
    vec_add_assert<<<BLOCKS, threadsPerBlock>>>(d_vec1, d_vec2, d_vec3, d_assert);

    // copy contents of d_assert to assert.
    cudaMemcpy(&assert, d_assert, sizeof(assert), cudaMemcpyDeviceToHost);

    // if assert == 0, print success. Else print failure.
    if(!assert){
        printf("assertions passed.\n");
    }
    else{
        printf("assertions failed.\n");
    }

    // free all device pointers from gpu memory.
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_vec3);
    cudaFree(d_assert);

    return 0;
}
