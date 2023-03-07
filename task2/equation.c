#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define BILLION 1000000000.0
#define MAX(x, y) (((x)>(y))?(x):(y))
#define ABS(x) ((x)<0 ? -(x):(x))
#define N 1024
int main(int argc, char *argv[]){
	struct timespec start, stop;
	clock_gettime(CLOCK_REALTIME, &start);
	double delta;
	int m;
	int iter_max;
	double tol;
	sscanf(argv[1], "%d", &m);
	sscanf(argv[2], "%d", &iter_max);
	sscanf(argv[3], "%f", &tol);
	int iter = 0;
        float err = tol + 1;
	float** arr = (float**)malloc(N * sizeof(float*));
	float** arrNew = (float**)malloc(N * sizeof(float*));
	for(int i = 0; i < N; i++)
	{
		arr[i] = (float*)malloc(N * sizeof(float));
		arrNew[i] = (float*)malloc(N * sizeof(float));
		for(int j = 0; j < m; j++)
		{
			arr[i][j] = 0;

		}
	}
	arr[0][0] = arrNew[0][0] = 10;
	arr[0][m-1] = arrNew[0][m-1] = 20;
	arr[m-1][0] = arrNew[m-1][0] = 30;
	arr[m-1][m-1] = arrNew[m-1][m-1] = 20; 
	float top, bottom, left, right; //coefficients for linear interpolation
	top = (arr[0][m-1] - arr[0][0])/(m-1);
	bottom = (arr[m-1][m-1] - arr[m-1][0])/(m-1);
	left = (arr[m-1][0] - arr[0][0])/(m-1);
	right = (arr[m-1][m-1] - arr[0][m-1])/(m-1);
	#pragma acc parallel loop
	for(int j = 1; j < m - 1; j++) { //linear interpolation
		arr[0][j] = (arr[0][0] + top*j);   //top
		arr[m-1][j] = (arr[m-1][0] + bottom*j); //bottom
		arr[j][0] = (arr[0][0] + left*j); //left
		arr[j][m-1] = (arr[0][m-1] + right*j); //right
	}

	//#pragma acc data copy(arr[:N][:N]) create(arrNew[:N][:N]) 
	//{
	while(err > tol && iter < iter_max) {
		err = 0;
		#pragma acc parallel loop reduction(max:err) 
		for(int j = 1; j < m - 1; j++)	{
			#pragma acc loop reduction(max:err) 
			for(int i = 1; i < m - 1; i++) {
				arrNew[i][j] = 0.25 * (arr[i+1][j] + arr[i-1][j]
					       	+ arr[i][j-1] + arr[i][j+1]); //calculating new value for the cell
				err = MAX(err, ABS((arrNew[i][j] - arr[i][j]))); //checking how much we improved
			}
		}
		#pragma acc parallel loop
		for (int j = 1; j < m - 1; j++) { //copying from the new array to the old array
			#pragma acc loop
			for (int i = 1; i < m - 1; i++) {
				arr[j][i] = arrNew[j][i];
			}
		}
		iter++;
	//}
	}
	clock_gettime(CLOCK_REALTIME, &stop);
        delta = (stop.tv_sec - start.tv_sec)
                +(double)(stop.tv_nsec - start.tv_nsec)/(double)BILLION;
        printf("Elapsed time %lf\n", delta);
	printf("Final result: %d, %0.6lf\n", iter, err);
	
	for (int i = 0; i < N; i++) {
		free(arr[i]);
		free(arrNew[i]);
	}
	free(arr);
	free(arrNew);
	return 0;
}

