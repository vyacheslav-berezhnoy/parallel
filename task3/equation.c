#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define BILLION 1000000000.0
#define MAX(x, y) (((x)>(y))?(x):(y))
#define ABS(x) ((x)<0 ? -(x): (x))
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
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
	if (m < 0 || m > 1024) {
		fprintf(stderr, "Not a valid grid size! It should be between 0 and 1024");
	}
	if (iter_max< 0 || iter_max > 1000000) {
                fprintf(stderr, "Not a valid number of iterations! It should be between 0 and 1000000");
        }
        int iter = 0;
        float err = tol + 1;
        float *arr = (float*)malloc(2 * m * m * sizeof(float));
        for(int j = 1; j <= m; j++)
        {
                for(int i = 1; i <= m; i++)
                {
                        arr[IDX2F(i,j,m)] = 0;

                }
        }
        arr[IDX2F(1,1,m)] = arr[IDX2F(1,m+1,m)] = 10;
        arr[IDX2F(1,m,m)] = arr[IDX2F(1,2*m,m)] = 20;
        arr[IDX2F(m,1,m)] = arr[IDX2F(m,m+1,m)] = 20;
        arr[IDX2F(m,m,m)] = arr[IDX2F(m,2*m,m)] = 30;
        /* coefficients for linear interpolation */
        float top, bottom, left, right;
        top = (arr[IDX2F(1,m,m)] - arr[IDX2F(1,1,m)])/(m-1);
        bottom = (arr[IDX2F(m,m,m)] - arr[IDX2F(m,1,m)])/(m-1);
        left = (arr[IDX2F(m,1,m)] - arr[IDX2F(1,1,m)])/(m-1);
        right = (arr[IDX2F(m,m,m)] - arr[IDX2F(1,m,m)])/(m-1);
	printf("\n");
        //linear interpolation
        for(int j = 1; j < m; j++) {
                arr[IDX2F(1,j,m)] = arr[IDX2F(1,j+m,m)] = (arr[IDX2F(1,1,m)] + top*(j-1));   //top
                arr[IDX2F(m,j,m)] = arr[IDX2F(m,j+m,m)]  = (arr[IDX2F(m,1,m)] + bottom*(j-1)); //bottom
                arr[IDX2F(j,1,m)]  = arr[IDX2F(j,m+1,m)] = (arr[IDX2F(1,1,m)] + left*(j-1)); //left
                arr[IDX2F(j,m,m)] = arr[IDX2F(j,2*m,m)] = (arr[IDX2F(1,m,m)] + right*(j-1)); //right
        }
	if (m == 15) {
	for (int i = 1; i <= m; i++) {
		for (int j = 1; j <= 2*m; j++) {
			printf("%06.3f ", arr[IDX2F(i,j,m)]);
		}
		printf("\n");
	}
	}
	cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
        	printf ("CUBLAS initialization failed\n");
        	return EXIT_FAILURE;
    	}
	float alpha = -1;
	int p, q;
	p = m;
	q = 0;
	int idx = 0;
	int flag = 1;
	float *buff = (float*)malloc(sizeof(float));
	#pragma acc data copy(arr[:(2*m*m)]) copyin(alpha, tol, top, bottom, left, right)
        {
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
                for(int j = 2; j < m; j++)  {
                        #pragma acc loop
                        for(int i = 2; i < m; i++) {
                                //calculating the new value for the cell
                                arr[IDX2F(i,j+p,m)] = 0.25 * (arr[IDX2F(i+1,j+q,m)] + arr[IDX2F(i-1,j+q,m)]
                                                + arr[IDX2F(i,j-1+q,m)] + arr[IDX2F(i,j+1+q,m)]);
                        }
                }
                if (!(iter%100)) {
                        err = 0;
			idx = 0;
			#pragma acc wait
			#pragma acc host_data use_device(arr)
			{
				//here we subtract arrays to find errors
				stat = cublasSaxpy(handle, m*m, &alpha, (arr+p*m), 1, (arr+q*m), 1);
				if (stat != CUBLAS_STATUS_SUCCESS) {
                                        printf ("Failed to subtract the arrays\n");
                                        // return EXIT_FAILURE;
                                }
				//now we find the index of the cell with the largest error
				stat = cublasIsamax(handle, m*m, (arr+q*m), 1, &idx);
				if (stat != CUBLAS_STATUS_SUCCESS) {
                                        printf ("Failed to find the max\n");
                                        // return EXIT_FAILURE;
                                }
				//
				stat = cublasGetVector(1, sizeof(*arr), (arr + q*m + idx - 1), 1, buff, 1);
				err = ABS(*buff);
				//if the error became lower than the tolerance, we exit
                                flag = err > tol;

			}
			//here we refill the borders, which were changed to zeros in cublasSaxpy
			#pragma acc parallel
			{
			arr[IDX2F(1,1,m)] = arr[IDX2F(1,m+1,m)] = 10;
                        arr[IDX2F(1,m,m)] = arr[IDX2F(1,2*m,m)] = 20;
                        arr[IDX2F(m,1,m)] = arr[IDX2F(m,m+1,m)] = 20;
                        arr[IDX2F(m,m,m)] = arr[IDX2F(m,2*m,m)] = 30;
			}
                        /* coefficients for linear interpolation */
			#pragma acc parallel loop
                        for(int j = 1; j < m; j++) {
                        	arr[IDX2F(1,j,m)] = arr[IDX2F(1,j+m,m)] = (arr[IDX2F(1,1,m)] + top*(j-1));   //top
                                arr[IDX2F(m,j,m)] = arr[IDX2F(m,j+m,m)]  = (arr[IDX2F(m,1,m)] + bottom*(j-1)); //bottom
                                arr[IDX2F(j,1,m)]  = arr[IDX2F(j,m+1,m)] = (arr[IDX2F(1,1,m)] + left*(j-1)); //left
                                arr[IDX2F(j,m,m)] = arr[IDX2F(j,2*m,m)] = (arr[IDX2F(1,m,m)] + right*(j-1)); //right
                        }

                }
                iter++;

        }
        }
        clock_gettime(CLOCK_REALTIME, &stop);
        delta = (stop.tv_sec - start.tv_sec)
                + (double)(stop.tv_nsec - start.tv_nsec)/(double)BILLION;
        printf("Elapsed time %lf\n", delta);
        printf("Final result: %d, %0.8f\n", iter, err);
	if (m == 15) {
        for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= 2*m; j++) {
                        printf("%06.3f ", arr[IDX2F(i,j,m)]);
                }
                printf("\n");
        }
        }
	cublasDestroy(handle);
        free(arr);
        return EXIT_SUCCESS;



}

