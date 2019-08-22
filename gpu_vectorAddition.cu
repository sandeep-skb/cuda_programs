#include <stdio.h>
#include <iostream>
#include <ctime>
using namespace std;

void host_init(float *arr, int n, float init_val){
	for (int i=0; i < n; i++)
		arr[i] = init_val;
}


void host_vec_add(float *h_a, float *h_b, float *h_c, int n){
	for (int i = 0; i < n; i++){
		h_c[i] = h_a[i] + h_b[i];
	}
}

__global__
void cuda_vec_add(float *d_a, float *d_b, float *d_c, int n){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n){
		d_c[i] = d_a[i] + d_b[i];
	}
}

void cudaError_check(cudaError_t err){
	if (err != cudaSuccess){
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
	}
}

int main(){
	float *h_a, *h_b, *h_c, *h_c_cpy;
	float *d_a, *d_b, *d_c;
	int n = 1024 * 1024 * 1024;
	size_t size = n*sizeof(float);
	clock_t start, stop;

//################## HOST Start ###################//
	h_a = (float*) malloc (size);
	h_b = (float*) malloc (size);
	h_c = (float*) malloc (size);
	h_c_cpy = (float*) malloc (size);


	host_init(h_a, n, 1);
	host_init(h_b, n, 2);
	host_init(h_c, n, 0);
	host_init(h_c_cpy, n, 0);
	start = clock();
	host_vec_add(h_a, h_b, h_c, n);
	stop = clock();
	double cpu_duration = (stop - start) / (double) CLOCKS_PER_SEC;
//################## HOST End ###################//

//################## CUDA Start ###################//
	cudaError_t err ;
	err = cudaMalloc((void **) &d_a, size);
	cudaError_check(err);
	err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaError_check(err);
	err = cudaMalloc((void **) &d_b, size);
	cudaError_check(err);
	err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	cudaError_check(err);
	err = cudaMalloc((void **) &d_c, size);
	cudaError_check(err);

	//Kernel invocation
	int num_threads_per_block = 256;
	dim3 gridDim  (n/num_threads_per_block, 1, 1);
	dim3 blockDim  (num_threads_per_block, 1, 1);
	start = clock();
	cuda_vec_add<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
	err = cudaMemcpy(h_c_cpy, d_c, size, cudaMemcpyDeviceToHost);
	cudaError_check(err);
	stop = clock();
	double gpu_duration = (stop - start) / (double) CLOCKS_PER_SEC;
//################## CUDA End ###################//
	int success = 1;
	for (int i = 0; i < n; i++){
		if (h_c[i] != h_c_cpy[i]){
			success = 0;
			printf("Failure at idx: %d\n", i);
			break;
		}
	}
	if (success == 1)
		printf("Success\n");
	printf("CPU Duration: %0.3f secs \n", cpu_duration);
	printf("GPU Duration: %0.3f secs \n", gpu_duration);
	return 1;
}
