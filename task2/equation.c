
	#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define BILLION 1000000000.0
#define MAX(x, y) (((x)>(y))?(x):(y))
#define ABS(x) ((x)<0 ? -(x): (x))
#define N 1024
int main(int argc, char *argv[]){
        struct timespec start, stop;
        clock_gettime(CLOCK_REALTIME, &start);
        double delta;
        int m;
        int iter_max;
        float tol;
        sscanf(argv[1], "%d", &m);
        sscanf(argv[2], "%d", &iter_max);
        sscanf(argv[3], "%f", &tol);
        int iter = 0;
        float err = tol + 1;
        float** arr = (float**)malloc(N * sizeof(float*));
        #pragma acc loop
        for(int i = 0; i < N; i++)
        {
                arr[i] = (float*)malloc(2*N * sizeof(float));
                for(int j = 0; j < m; j++)
                {
                        arr[i][j] = 0;

                }
        }
        arr[0][0] = arr[0][m] = 10;
        arr[0][m-1] = arr[0][2*m-1] = 20;
        arr[m-1][0] = arr[m-1][m] = 30;
        arr[m-1][m-1] = arr[m-1][2*m-1] = 20;
        /* coefficients for linear interpolation */
        float top, bottom, left, right;
        top = (arr[0][m-1] - arr[0][0])/(m-1);
        bottom = (arr[m-1][m-1] - arr[m-1][0])/(m-1);
        left = (arr[m-1][0] - arr[0][0])/(m-1);
        right = (arr[m-1][m-1] - arr[0][m-1])/(m-1);
        #pragma acc parallel loop
        //linear interpolation
        for(int j = 1; j < m - 1; j++) {
                arr[0][j] = arr[0][j+m] = (arr[0][0] + top*j);   //top
                arr[m-1][j] = arr[m-1][j+m]  = (arr[m-1][0] + bottom*j); //bottom
                arr[j][0] = arr[j][m] = (arr[0][0] + left*j); //left
                arr[j][m-1] = arr[j][2*m-1] = (arr[0][m-1] + right*j); //right
        }
	#pragma acc data copy(arr[:m][:(2*m)])
        {
        int p, q;
        int flag = 1;
        while(iter < iter_max && flag) {
                if (iter % 2) { //here we change what part of the array
                                //is considered new
                        p = 0;
                        q = m;
                }
                else {
                        p = m;
                        q = 0;
                }
                #pragma acc parallel loop
                for(int j = 1; j < m - 1; j++)  {
                        #pragma acc loop
                        for(int i = 1; i < m - 1; i++) {
                                //calculating the new value for the cell
                                arr[i][j+p] = 0.25 * (arr[i+1][j+q] + arr[i-1][j+q]
                                                + arr[i][j-1+q] + arr[i][j+1+q]);
                                //err = MAX(err, ABS((arr[i][j+m] - arr[i][j])));
                        }
                }
                if (!(iter%100)) {
                        err = 0;
                        #pragma acc parallel loop reduction(max:err)
                        for(int j = 1; j < m - 1; j++) {
                                #pragma acc loop reduction(max:err)
                                for(int i = 1; i < m - 1; i++) {
                                        err = MAX(err, ABS((arr[i][j+m] - arr[i][j])));
                                }
                        }
                        flag = err > tol;
                }
                iter++;

        }
        }
        clock_gettime(CLOCK_REALTIME, &stop);
        delta = (stop.tv_sec - start.tv_sec)
                +(double)(stop.tv_nsec - start.tv_nsec)/(double)BILLION;
        printf("Elapsed time %lf\n", delta);
        printf("Final result: %d, %0.6f\n", iter, err);

        #pragma acc parallel loop
        for (int i = 0; i < N; i++) {
                free(arr[i]);
        }
        free(arr);
        return 0;



}

