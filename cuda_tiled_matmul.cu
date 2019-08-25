#include <stdio.h>
#include <iostream>
#include <ctime>
using namespace std;

#define TILE_WIDTH 16
void host_init(float *arr, int n, float init_val){
	for (int i=0; i < n; i++)
		arr[i] = init_val;
}


void host_matmul(float *h_a, float *h_b, float *h_c, int m, int n, int k){
	// h_a (mxk); h_b (kxn); h_c (mxn)
	for (int row = 0; row < m; row++){
		for (int col = 0; col < n; col++){
			float sum = 0;
			for (int kk = 0; kk < k; kk++){
				float a = h_a[row*k + kk];
				float b = h_b[kk*n + col];
				sum += a * b;
			}
			h_c[row*n + col] = sum;
		}
	}
}

__global__
void cuda_matmul(float *d_a, float *d_b, float *d_c, int m, int n, int k){
	__shared__ float shmem_a[TILE_WIDTH][TILE_WIDTH];
	__shared__ float shmem_b[TILE_WIDTH][TILE_WIDTH];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x*blockDim.x + tx;
	int row = blockIdx.y*blockDim.y + ty;
	float c_val = 0.0;
	for (int phase = 0; phase < (k-1)/TILE_WIDTH + 1; phase++){
		if (row < m && (tx + phase*TILE_WIDTH) < k)
			shmem_a[ty][tx] = d_a[row*k + (tx + phase*TILE_WIDTH)];
		else
			shmem_a[ty][tx] = 0.0;

		if (col < n && ((ty + phase*TILE_WIDTH) < k))
			shmem_b[ty][tx] = d_b[(ty + phase*TILE_WIDTH)*n + col];
		else
			shmem_b[ty][tx] = 0.0;


		__syncthreads();
		for (int kk = 0; kk < TILE_WIDTH; kk++){
			c_val += shmem_a[ty][kk] * shmem_b[kk][tx];
		}
		__syncthreads();
	}
	if (col < n and row < m)
		d_c[row*n + col] = c_val;
}

void cudaError_check(cudaError_t err, int line){
	if (err != cudaSuccess){
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(err), __FILE__, line);
		exit(EXIT_FAILURE);
	}
}

int main(){
	float *h_a, *h_b, *h_c, *h_c_cpy;
	float *d_a, *d_b, *d_c;
	// h_a dimensions = MxK
	// h_b dimensions = KxN
	// h_c dimensions = MxN
	int m = 1024; // 
	int n = 1024; // 
	int k = 1024; //

	size_t size_ha = k*m*sizeof(float);
	size_t size_hb = k*n*sizeof(float);
	size_t size_hc = m*n*sizeof(float);
	clock_t start, stop;

//################## HOST Start ###################//
	h_a = (float*) malloc (size_ha);
	h_b = (float*) malloc (size_hb);
	h_c = (float*) malloc (size_hc);
	h_c_cpy = (float*) malloc (size_hc);


	host_init(h_a, k*m, 1);
	host_init(h_b, n*k, 2);
	host_init(h_c, n*m, 0);
	host_init(h_c_cpy, n*m, 0);
	start = clock();
	host_matmul(h_a, h_b, h_c, m, n, k);
	stop = clock();
	double cpu_duration = (stop - start) / (double) CLOCKS_PER_SEC;
//################## HOST End ###################//

//################## CUDA Start ###################//
	cudaError_t err ;
	err = cudaMalloc((void **) &d_a, size_ha);
	cudaError_check(err, __LINE__);
	err = cudaMemcpy(d_a, h_a, size_ha, cudaMemcpyHostToDevice);
	cudaError_check(err, __LINE__);
	err = cudaMalloc((void **) &d_b, size_hb);
	cudaError_check(err, __LINE__);
	err = cudaMemcpy(d_b, h_b, size_hb, cudaMemcpyHostToDevice);
	cudaError_check(err, __LINE__);
	err = cudaMalloc((void **) &d_c, size_hc);
	cudaError_check(err, __LINE__);

	//Kernel invocation
	int num_threads_per_block = TILE_WIDTH;
	dim3 gridDim  ((m-1)/num_threads_per_block + 1, (n-1)/num_threads_per_block + 1, 1);
	dim3 blockDim  (num_threads_per_block, num_threads_per_block, 1);
	start = clock();
	cuda_matmul<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);
	stop = clock();
	err = cudaMemcpy(h_c_cpy, d_c, size_hc, cudaMemcpyDeviceToHost);
	cudaError_check(err, __LINE__);
	double gpu_duration = (stop - start) / (double) CLOCKS_PER_SEC;
//################## CUDA End ###################//
	int success = 1;
	for (int i = 0; i < n*m; i++){
		if (h_c[i] != h_c_cpy[i]){
			success = 0;
			printf("Failure at idx: %d\n", i);
			break;
		}
	}
	if (success == 1)
		printf("Success\n");
	printf("CPU Duration: %0.3f secs \n", cpu_duration);
	printf("GPU Duration: %0.5f secs \n", gpu_duration);
	return 1;
}
