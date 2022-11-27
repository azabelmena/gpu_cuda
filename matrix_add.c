// Program that adds the entries of a matrix.

#include<stdio.h>
#include<stdlib.h>

// set default size to 10 (work with 10 x 10 matrices)
const int SIZE = 10;

// populate the matrices in the arguments.
void mat_populate(int mat1[][SIZE], int mat2[][SIZE], int mat3[][SIZE]){
    for(int i = 0; i < SIZE ; i++){
        for(int j = 0; j < SIZE ; j++){
            mat1[i][j] = i+j; // add i and j for the ij-th entry.
            mat2[i][j] = i*j; // multiply i and j for the ij-th entry.
            mat3[i][j] = 0; // set all entries to 0.
        }
    }

    return;
}

// add the entries of the first two arguments and store them in the third
// argument.
void mat_add(int mat1[][SIZE], int mat2[][SIZE], int mat3[][SIZE]){
    for(int i = 0; i < SIZE ; i++){
        for(int j = 0; j < SIZE ; j++){
            mat3[i][j] = mat1[i][j]+mat2[i][j];
        }
    }

    return;
}

// Verify that the addition was performed correctly.
// return 0: the addition was performed succesfully.
// return 1: addition error at entry ij.
int mat_add_assert(int mat1[][SIZE], int mat2[][SIZE], int mat3[][SIZE]){
    int assert = 0;

    for(int i = 0; i < SIZE ; i++){
        for(int j = 0; j < SIZE ; j++){
            if(!(mat3[i][j] == mat1[i][j]+mat2[i][j])){
                assert = 1;

                return assert;
            }
        }
    }

    return assert;
}

// print the entries of the matrix in matrix format
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

    // declare the matrices of dimensions SIZE x SIZE.
    int mat1[SIZE][SIZE];
    int mat2[SIZE][SIZE];
    int mat3[SIZE][SIZE];

    // populate mat1 mat2, and mat3; add mat1 and mat2, store in mat3.
    mat_populate(mat1, mat2, mat3);
    mat_add(mat1, mat2, mat3);

    // grab assertion from the matrix assertion method.
    int assert = mat_add_assert(mat1, mat2, mat3);

    // Check assert, if 0, print passed, else print failed.
    if(!assert){
        printf("Assertions passed.\n");
    }
    else{
        printf("Assertions failed.\n");
    }

    return 0;
}
