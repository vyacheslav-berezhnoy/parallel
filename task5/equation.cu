#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <utility>
#define BILLION 1000000000.0

#define checkMPIErrors(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (MPI_SUCCESS != mpi_status) {                                              \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
            exit( mpi_status );                                                       \
        }                                                                             \
    }

#define checkCudaErrors(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit( cudaStatus );                                                             \
        }                                                                                   \
    }

__global__ void getAverage(const double *a, double *a_new, 
                          int m, int iy_start, int iy_end) {

  //we assign the cell the average value from a cross surrounding it

  int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
  int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

  if (iy < iy_end && ix < (m - 1)) {
      a_new[iy * m + ix] = 0.25 * (a[iy * m + ix + 1] + a[iy * m + ix - 1] +
                                   a[(iy + 1) * m + ix] + a[(iy - 1) * m + ix]);
  }

}
__global__ void subtractArrays(const double *a, const double *a_new,
		double *err_arr,
                          int m, int iy_start, int iy_end) {

  int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
  int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

  if (iy < iy_end  && ix < (m-1)) {
	  err_arr[iy*m + ix]  = std::fabs(a[iy * m + ix] -  a_new[iy * m + ix]);
  }
}
void launch_jacobi_kernel(const double *a, double *a_new,
                int m, int iy_start, int iy_end, cudaStream_t stream) {

	/*

        const int grid_size_x = (m + 32 - 1)/32;
        const int grid_size_y = ((iy_end - iy_start) + 32 - 1) / 32;
        dim3 grid(grid_size_x, grid_size_y);
        dim3 block(32, 32);
        getAverage<<<grid, block, 0, stream>>>(a, a_new, m, iy_start, iy_end);
	*/
	int dim_block_x = 32;
    	int dim_block_y = 32;
    	dim3 dim_grid((m + dim_block_x - 1) / dim_block_x,
                  ((iy_end - iy_start) + dim_block_y - 1) / dim_block_y, 1);
	getAverage<<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, stream>>>(a, a_new, m, iy_start, iy_end);
	checkCudaErrors(cudaGetLastError());
}

int main(int argc, char *argv[]){

        int m;
        int iter_max;
        double tol;
        sscanf(argv[1], "%d", &m);
        sscanf(argv[2], "%d", &iter_max);
        sscanf(argv[3], "%lf", &tol);
	if (m < 0 || m > 8000) {
		fprintf(stderr, "Not a valid grid size! It should be between 0 and 8000");
		exit(EXIT_FAILURE);
	}
	if (iter_max < 0 || iter_max > 1000000) {
                fprintf(stderr, "Not a valid number of iterations! It should be between 0 and 1000000");
		exit(EXIT_FAILURE);
        }
        int iter = 0;
	double *err = NULL;
    	cudaMallocHost(&err, sizeof(double));
        (*err) = tol + 1;

	// MPI initialization
    	int rank, group_size;
    	checkMPIErrors(MPI_Init(&argc, &argv));
    	checkMPIErrors(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    	checkMPIErrors(MPI_Comm_size(MPI_COMM_WORLD, &group_size));
	checkCudaErrors(cudaSetDevice(rank));
	size_t size = m * m * sizeof(double);
        //double *arr = (double*)malloc(size);
	double *arr = NULL;
	checkCudaErrors(cudaMallocHost(&arr, size));
	checkCudaErrors(cudaMemset(arr, 0, size));

        arr[0] = 10;
        arr[m - 1] = 20;
        arr[m*m - 1] = 30;
        arr[m*(m-1)] = 20;

        /* coefficients for linear interpolation */
	double top, bottom, left, right;
	top = (arr[m-1] - arr[0])/(m-1);
        bottom = (arr[m*m - 1] - arr[m*(m-1)])/(m-1);
        left = (arr[m*(m-1)] - arr[0])/(m-1);
        right = (arr[m*m - 1] - arr[m-1])/(m-1);


	//linear interpolation
	for(int j = 1; j < m - 1; j++) {
                arr[j] = (arr[0] + top*j);   //top
                arr[m*(m-1) + j] = (arr[m*(m-1)] + bottom*j); //bottom
                arr[m*j] = (arr[0] + left*j); //left
                arr[m*j + m - 1] = (arr[m-1] + right*j); //right
        }

	int chunk_size;
    	int chunk_size_low = (m - 2) / group_size;
    	int chunk_size_high = chunk_size_low + 1;
    	int num_ranks_low = group_size * chunk_size_low + group_size -
                        	(m - 2);  // Number of ranks with chunk_size = chunk_size_low
    	if (rank < num_ranks_low) {
        	chunk_size = chunk_size_low;
	}
    	else {
        	chunk_size = chunk_size_high;
	}
	int iy_start_global;  // My start index in the global array
    	if (rank < num_ranks_low) {
        	iy_start_global = rank * chunk_size_low + 1;
    	} else {
        	iy_start_global =
            	num_ranks_low * chunk_size_low + (rank - num_ranks_low) * chunk_size_high + 1;
    	}
    	int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array

    	int iy_start = 1;
    	int iy_end = iy_start + chunk_size;
	double *a = NULL;
  	checkCudaErrors(cudaMalloc((void **)&a, m * (chunk_size + 2) * sizeof(double)));
	double *a_new = NULL;
	checkCudaErrors(cudaMalloc((void **)&a_new, m * (chunk_size + 2) * sizeof(double)));
	double *err_arr = NULL;
        checkCudaErrors(cudaMalloc((void **)&err_arr, m * (chunk_size + 2) * sizeof(double)));
	checkCudaErrors(cudaMemset(err_arr, 0, m * (chunk_size + 2) * sizeof(double)));
	cudaStream_t top_stream, compute_stream, bottom_stream;
        checkCudaErrors(cudaStreamCreate(&top_stream));
	checkCudaErrors(cudaStreamCreate(&compute_stream));
	checkCudaErrors(cudaStreamCreate(&bottom_stream));
	checkCudaErrors(cudaMemcpy(a, arr + m * (iy_start_global-1), m * (chunk_size + 2) * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(a_new, arr + m * (iy_start_global-1), m * (chunk_size + 2) * sizeof(double), cudaMemcpyHostToDevice));

	double *d_err = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_err, sizeof(double)));


	double *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
	//we call DeviceReduce here to check how much memory we need for temporary storage
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, err_arr, d_err, m*(chunk_size+2));
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
	nvtxRangePushA("Main loop");
    	clock_t begin = clock();
        while(iter < iter_max && (*err) > tol) {
		
		//launch_jacobi_kernel( a, a_new, iy_start, (iy_start+1), m, top_stream );
		//launch_jacobi_kernel( a, a_new, (iy_end-1), iy_end, m, bottom_stream );
		//launch_jacobi_kernel( a, a_new, (iy_start+1), (iy_end-1), m, compute_stream );
		{
			//top
		int dim_block_x = 32;
        	int dim_block_y = 32;
        	dim3 dim_grid((m + dim_block_x - 1) / dim_block_x,
                  	(((iy_start+1) - iy_start) + dim_block_y - 1) / dim_block_y, 1);
        	getAverage<<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, top_stream>>>(a, a_new, m, iy_start, (iy_start+1));
		}
		{
			//bottom
                int dim_block_x = 32;
                int dim_block_y = 32;
                dim3 dim_grid((m + dim_block_x - 1) / dim_block_x,
                        ((iy_end - (iy_end-1)) + dim_block_y - 1) / dim_block_y, 1);
                getAverage<<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, bottom_stream>>>(a, a_new, m, (iy_end-1), iy_end);
                }
		{
			//bulk
                int dim_block_x = 32;
                int dim_block_y = 32;
                dim3 dim_grid((m + dim_block_x - 1) / dim_block_x,
                        (((iy_end-1) - (iy_start+1)) + dim_block_y - 1) / dim_block_y, 1);
                getAverage<<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, compute_stream>>>(a, a_new, m, (iy_start+1), (iy_end-1));
                }
		cudaStreamSynchronize( top_stream );
		cudaStreamSynchronize( bottom_stream );
		if (rank != 0) {
			MPI_Sendrecv( a_new+iy_start*m, m, MPI_DOUBLE, rank - 1, 0,
				a_new, m, MPI_DOUBLE, rank - 1, 0,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE );
		}
		if (rank != group_size - 1) {

			MPI_Sendrecv( a_new+(iy_end-1)*m, m, MPI_DOUBLE, rank + 1, 0,
				a_new+(iy_end*m), m, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE );
		}
		
		if (iter % 100 == 0){
			const int grid_size_x = (m + 32 - 1)/32;
        		const int grid_size_y = ((iy_end - iy_start) + 32 - 1) / 32;
        		dim3 grid(grid_size_x, grid_size_y);
        		dim3 block(32, 32);
            		subtractArrays<<<grid, block, 0, compute_stream>>>(a_new, a, err_arr, m, iy_start, iy_end);
            		// find the maximum error
			cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, err_arr, d_err, m*(chunk_size+2), compute_stream);
            		// stream syncing
            	        checkCudaErrors(cudaStreamSynchronize(compute_stream));
            		checkMPIErrors(MPI_Allreduce(d_err, d_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
            		// copy to host memory
            		checkCudaErrors(cudaMemcpyAsync(err, d_err, sizeof(double), cudaMemcpyDeviceToHost, compute_stream));
		}
		std::swap(a_new, a);
		iter++;
        }

	clock_t end = clock();
    	nvtxRangePop();

	checkCudaErrors(cudaMemcpy(arr + m * (iy_start_global-1), a,  m * (chunk_size + 2) * sizeof(double), cudaMemcpyDeviceToHost));
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0 && m < 25) {
		std::cout << "RANK: " << rank << std::endl << std::endl;
		for (int i = iy_start_global - 1; i < iy_end_global + 2; i++) {
			for (int j = 0; j < m; j++) {
				printf("%06.3f ", arr[i*m + j]);
			}
			std::cout << std::endl;
		}
	}
	if (rank == 1 && m < 25) {
                std::cout << "RANK: " << rank << std::endl << std::endl;
                for (int i = iy_start_global - 1; i < iy_end_global + 2; i++) {
                        for (int j = 0; j < m; j++) {
                                printf("%06.3f ", arr[i*m + j]);
                        }
                        std::cout << std::endl;
                }
        }
	if (rank == 2 && m < 25) {
                std::cout << "RANK: " << rank << std::endl << std::endl;
                for (int i = iy_start_global - 1; i < iy_end_global + 2; i++) {
                        for (int j = 0; j < m; j++) {
                                printf("%06.3f ", arr[i*m + j]);
                        }
                        std::cout << std::endl;
                }
        }
	if (rank == 3 && m < 25) {
                std::cout << "RANK: " << rank << std::endl << std::endl;
                for (int i = iy_start_global - 1; i < iy_end_global + 2; i++) {
                        for (int j = 0; j < m; j++) {
                                printf("%06.3f ", arr[i*m + j]);
                        }
                        std::cout << std::endl;
                }
        }
       	if (rank == 0){
        	std::cout << "Error: " << (*err) << std::endl;
        	std::cout << "Iteration: " << iter << std::endl;
        	std::cout << "Time: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl;
    	}	
	checkCudaErrors(cudaFreeHost(arr));
	checkCudaErrors(cudaFree(d_err));
	checkCudaErrors(cudaFree(a));
	checkCudaErrors(cudaFree(a_new));
	checkCudaErrors(cudaFree(d_temp_storage));
	checkMPIErrors(MPI_Finalize());

        return 0;

}

