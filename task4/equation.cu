#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#define BILLION 1000000000.0
#define MAX(x, y) (((x)>(y))?(x):(y))
#define ABS(x) ((x)<0 ? -(x): (x))
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
__global__ void fillBorders(float *arr, float top,
		float bottom, float left, float right,
                          int m) {

  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if ((j > 0) && (j < m)) {
	  arr[IDX2F(1,j,m)] = arr[IDX2F(1,j+m,m)] = (arr[IDX2F(1,1,m)] + top*(j-1));   //top
          arr[IDX2F(m,j,m)] = arr[IDX2F(m,j+m,m)]  = (arr[IDX2F(m,1,m)] + bottom*(j-1)); //bottom
          arr[IDX2F(j,1,m)]  = arr[IDX2F(j,m+1,m)] = (arr[IDX2F(1,1,m)] + left*(j-1)); //left
          arr[IDX2F(j,m,m)] = arr[IDX2F(j,2*m,m)] = (arr[IDX2F(1,m,m)] + right*(j-1)); //right
  }
}
__global__ void getAverage(float *arr, int p, int q,
                          int m) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if ((i > 1) && (i < m) && (j > 1) && (j < m)) {
    arr[IDX2F(i,j+p,m)] = 0.25 * (arr[IDX2F(i+1,j+q,m)] + arr[IDX2F(i-1,j+q,m)]
                                                + arr[IDX2F(i,j-1+q,m)] + arr[IDX2F(i,j+1+q,m)]);
  }
}
__global__ void subtractArrays(const float *arr_a, float *arr_b,
                          int m) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if ((i > 1) && (i < m) && (j > 1) && (j < m)) {
	  arr_b[IDX2F(i,j,m)] = ABS(arr_a[IDX2F(i,j,m)] - arr_a[IDX2F(i,j+m,m)]);
  }
}
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
	size_t size = 2 * m * m * sizeof(float);
        float *arr = (float*)malloc(size);

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

	cudaError_t cudaErr = cudaSuccess;
	float *d_A = NULL;
  	cudaErr = cudaMalloc((void **)&d_A, size);
	if (cudaErr != cudaSuccess) {
    		fprintf(stderr,
            		"(error code %s)!\n",
            		cudaGetErrorString(cudaErr));
    		exit(EXIT_FAILURE);
  	}	
	float *d_B = NULL;
	cudaErr = cudaMalloc((void **)&d_B, size/2);
	cudaErr = cudaMemcpy(d_A, arr, size, cudaMemcpyHostToDevice);
	if (cudaErr != cudaSuccess) {
                fprintf(stderr,
                        "(error code %s)!\n",
                        cudaGetErrorString(cudaErr));
                exit(EXIT_FAILURE);
        }
	fillBorders<<<1, 1024>>>(d_A, top, bottom, left, right, m);
	cudaErr = cudaMemcpy(arr, d_A, size, cudaMemcpyDeviceToHost);
	if (cudaErr != cudaSuccess) {
                fprintf(stderr,
                        "(error code %s)!\n",
                        cudaGetErrorString(cudaErr));
                exit(EXIT_FAILURE);
        }
	printf("\n");
	if (m == 13) {
	for (int i = 1; i <= m; i++) {
		for (int j = 1; j <= 2*m; j++) {
			printf("%06.3f ", arr[IDX2F(i,j,m)]);
		}
		printf("\n");
	}
	}

	int p, q;
	p = m;
	q = 0;
	int flag = 1;
	float *h_buff = (float*)malloc(sizeof(float));
	float *d_buff = NULL;
	cudaErr = cudaMalloc((void**)&d_buff, sizeof(float));
	if (cudaErr != cudaSuccess) {
                fprintf(stderr,
                        "(error code %s)!\n",
                        cudaGetErrorString(cudaErr));
                exit(EXIT_FAILURE);
        }
	
	dim3 grid((m + 32 - 1)/32 , (m + 32 - 1)/32);

	dim3 block(32, 32);

	void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_B, d_buff, m*m);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
	bool graphCreated=false;
	cudaGraph_t graph;
	cudaGraphExec_t instance;
	cudaStream_t stream;
	cudaStreamCreate(&stream);

        {
        while(iter < iter_max && flag) {
		if(!graphCreated) {
			cudaErr = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

			if (cudaErr != cudaSuccess) {
               			 fprintf(stderr,
                        	"Failed to start stream capture (error code %s)!\n",
                        	cudaGetErrorString(cudaErr));
                		exit(EXIT_FAILURE);
        		}
			for (int i = 0; i < 100; i++) {
				q = (i % 2) * m;
				p = m - q;
				getAverage<<<grid, block, 0, stream>>>(d_A, p, q, m);
			}
			cudaErr = cudaStreamEndCapture(stream, &graph);
			if (cudaErr != cudaSuccess) {
                                 fprintf(stderr,
                                "Failed to end stream capture (error code %s)!\n",
                                cudaGetErrorString(cudaErr));
                                exit(EXIT_FAILURE);
                        }
			cudaErr = cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
			if (cudaErr != cudaSuccess) {
                                 fprintf(stderr,
                                "Failed to instantiate cuda graph (error code %s)!\n",
                                cudaGetErrorString(cudaErr));
                                exit(EXIT_FAILURE);
                        }
			graphCreated=true;

		}
		cudaErr = cudaGraphLaunch(instance, stream);
		if (cudaErr != cudaSuccess) {
                                 fprintf(stderr,
                                "Failed to launch cuda graph (error code %s)!\n",
                                cudaGetErrorString(cudaErr));
                                exit(EXIT_FAILURE);
                }
		cudaErr = cudaStreamSynchronize(stream);
		if (cudaErr != cudaSuccess) {
                                 fprintf(stderr,
                                "Failed to synchronize the stream (error code %s)!\n",
                                cudaGetErrorString(cudaErr));
                                exit(EXIT_FAILURE);
                }
		iter += 100;
		subtractArrays<<<grid, block, 0, stream>>>(d_A, d_B, m);
		cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_B, d_buff, m*m);
		cudaErr = cudaMemcpyAsync(h_buff, d_buff, sizeof(float), cudaMemcpyDeviceToHost, stream);
		if (cudaErr != cudaSuccess) {
                                 fprintf(stderr,
                                "?(error code %s)!\n",
                                cudaGetErrorString(cudaErr));
                                exit(EXIT_FAILURE);
                }
		err = *h_buff;
		flag = err > tol;
        }
        }

        clock_gettime(CLOCK_REALTIME, &stop);
        delta = (stop.tv_sec - start.tv_sec)
                + (double)(stop.tv_nsec - start.tv_nsec)/(double)BILLION;
        printf("Elapsed time %lf\n", delta);
        printf("Final result: %d, %0.8f\n", iter, err);
	cudaErr = cudaMemcpy(arr, d_A, size, cudaMemcpyDeviceToHost);

	if (m == 13) {
        for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= 2*m; j++) {
                        printf("%06.3f ", arr[IDX2F(i,j,m)]);
                }
                printf("\n");
        }
        }
        free(arr);
	free(h_buff);
	cudaFree(d_buff);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_temp_storage);
        return EXIT_SUCCESS;

}

