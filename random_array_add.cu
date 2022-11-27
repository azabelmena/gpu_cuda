#include<cuda.h>
#include<time.h>
#include<stdio.h>
#include<stdlib.h>

const int COUNT = 1 << 19;

void array_populate(int *arr1, int *arr2){

    for(int i = 0; i < COUNT; i++){
        arr1[i]  = rand() % COUNT;
        arr2[i]  = rand() % COUNT;
    }

    return;
}

void arr_print(int *arr){

    for(int i = 0; i < COUNT ; i++){
        printf("%d\t", arr[i]);
    }
    printf("\n");

    return;
}

__global__ void array_add(int *arr1, int *arr2){

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < COUNT){
        arr1[id] += arr2[id];
    }

    return;
}

int main(){

    srand(time(0));
    int *h_a = new int[COUNT];
    int *h_b = new int[COUNT];

    array_populate(h_a,h_b);

    int *d_a = 0;
    int *d_b = 0;

    cudaMalloc(&d_a, sizeof(h_a));

    cudaMalloc(&d_b, sizeof(h_b));

    cudaMemcpy(d_a, &h_a, sizeof(h_a), cudaMemcpyHostToDevice);

    array_add<<<(COUNT/256+1),256>>>(d_a, d_b);

    cudaMemcpy(&h_a, d_a, sizeof(h_a), cudaMemcpyDeviceToHost);

    arr_print(h_a);

    cudaFree(d_a);
    cudaFree(d_b);

    free(h_a);
    free(h_b);

    return 0;
}
