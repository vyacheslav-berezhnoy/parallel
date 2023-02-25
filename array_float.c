#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define N 10000000
const double angle = (2 * M_PI) / N;
int main() {
        float *arr = (float*)malloc(sizeof(float)*N);
	float sum;
	sum = 0;
	struct timespec start, end;
        clock_gettime(CLOCK_REALTIME, &start);
        #pragma acc data create(arr[:N]) copyin(angle) copy(sum)
	{
        #pragma acc kernels
        {
        for (int i = 0; i < N; i++) {
		arr[i] = sinf(angle * i);
        }
        }
	#pragma acc kernels
        {
        for (int i = 0; i < N; i++) {
                sum += arr[i];
        }
	}
	}
	clock_gettime(CLOCK_REALTIME, &end);
        double time_spent = (end.tv_nsec - start.tv_nsec)/1000.0;
        printf("time %f\n", time_spent);
        printf("%.9g\n", sum);
        free(arr);
        return 0;
}

