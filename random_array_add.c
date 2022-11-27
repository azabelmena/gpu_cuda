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

int main(){

    srand(time(0));
    int *h_a = malloc(COUNT*sizeof(int));
    int *h_b = malloc(COUNT*sizeof(int));

    array_populate(h_a, h_b);
    arr_print(h_a);
    arr_print(h_b);

    return 0;
}
