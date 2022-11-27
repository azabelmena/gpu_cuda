#include<stdio.h>
#include<stdlib.h>

// Set size to 2**8=256
const int SIZE = 131072;

void hello_world(){
    for(int i = 0; i < SIZE ; i++){
        printf("Hello world from iteration %d\n", i);
    }
}

int main(){

    hello_world();

    return 0;
}
