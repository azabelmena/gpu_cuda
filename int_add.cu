#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>

const int SIZE = 1 << 30;

// Add two integers and store it into a third integer.
__global__ void int_add(int *a, int *b, int *c){

    int i = threadIdx.x;

    if(i < SIZE){
        c[0] = a[0]+b[0];
    }

    return;
}

int main(){

    int a = 5;
    int b = 9;
    int c = 0;

    int *d_a = 0;
    int *d_c = 0;
    int *d_b = 0;

    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_c, sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice);

    int_add<<<64, 64>>>(d_a,d_b,d_c);

    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d+%d=%d\n",a,b,c);

    return 0;
}
