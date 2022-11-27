#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<stdlib.h>

const int SIZE = 1024;

__global__ void hello_world(int size){

    printf("Hello world from block %d, thread %d.\n", blockIdx.x, threadIdx.x);

    return;
}

int main(){

    hello_world<<<SIZE, SIZE>>>(SIZE);
    cudaDeviceSynchronize();

    return 0;
}
