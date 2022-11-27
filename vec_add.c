// Program that adds two vectors (arrays) using the cpu.

#include<stdio.h>
#include<stdlib.h>

const int SIZE = (1 << 19);

// Populate the vectors (arrays) according to assigment parameters.
void vec_populate(int *vec1, int *vec2, int *vec3){

    for(int i = 0; i < SIZE ; i++){
        vec1[i] = i*i; // populate vec1 with entries of 1 shifted by i bytes.
        vec2[i] = i*i*i; // populate vec2 with entries of i shifted by 1 bytes.
        vec3[i] = 0; // populate vec3 with 0s.
    }

    return;
}

// Add the the first two vector (array) arguments and store it into the third
// vector argument.
void vec_add(int *vec1, int *vec2, int *vec3){

    for(int i = 0; i < SIZE ; i++){
        vec3[i] = vec1[i]+vec2[i];
    }

    return;
}

int vec_add_assert(int *vec1, int *vec2, int *vec3){

    int assert = 0;

    for(int i = 0; i < SIZE ; i++){
        if(!(vec3[i] == vec1[i]+vec2[i])){
            assert = 1;

            return assert;
        }
    }

    return assert;
}

void vec_print(int *vec){

    for(int i = 0; i < SIZE; i++){
        printf("%d\t", vec[i]);
    }
    printf("\n");

    return;
}

int main(){

    // Initialize and allocate memory for vec1, vec2, vec3.
    int *vec1 = malloc(SIZE*sizeof(int));
    int *vec2 = malloc(SIZE*sizeof(int));
    int *vec3 = malloc(SIZE*sizeof(int));

    vec_populate(vec1, vec2, vec3);

    // Populate and add the vectors.
    vec_add(vec1, vec2, vec3);

    int assert = vec_add_assert(vec1, vec2, vec3);

    if(!assert){
        printf("Assertions passed.\n");
    }
    else{
        printf("Assertions failed.\n");

        return 2;
    }

    return 0;
}
