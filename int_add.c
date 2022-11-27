#include<stdio.h>
#include<stdlib.h>

const int SIZE = 1 << 30;

// Add two integers and store it into a third integer.
void int_add(int *a, int *b, int *c){

    for(int i = 0; i < SIZE ; i++){
        c[0] = a[0]+b[0];
    }

    return;
}

int main(){

    int a = 5;
    int b = 9;
    int c = 0;

    int_add(&a,&b,&c);

    printf("%d+%d=%d\n", a,b,c);

    return 0;
}
