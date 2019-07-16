#include <stdio.h>
#include <stdlib.h>



__global__
void gpu_conv1d(float *d_out, float *d_in, float *d_filter, int size_in, int size_filter){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float sum = 0.0;
	int offset = size_filter / 2;
	if (i < size_in){
		for (int j=0; j < size_filter; j++){
			if ((i-offset+j) >= 0 && (i - offset + j) <= size_in)
				sum += d_in[i-offset+j]*d_filter[j];
		}
		d_out[i] = sum;
	}

}

void init(float *arr, int N, float val){
	for (int r=0; r < N; r++){
			arr[r] = val;
	}
}

void host_conv1d(float *h_out, float *h_in, float *h_filter, int size_in, int size_filter){
	
	int offset = size_filter / 2;
	for (int i =0; i < size_in; i++){
		float sum = 0.0;
		for (int j = 0; j < size_filter; j++){
			if ((i-offset+j) >= 0 && (i - offset + j) <= size_in)
				sum += h_in[i-offset+j]*h_filter[j];
		}
		h_out[i] = sum;
	}

}

int main(){
	float *d_in, *d_filter, *d_out;
	float *h_in, *h_filter, *h_out; // Only for checking. Not needed for functionality


	int din_size = 24;
	int filter_size = 5;

	size_t size_in = din_size*sizeof(float);
	size_t size_filter = filter_size*sizeof(float);
	size_t size_out = din_size*sizeof(float);

	h_in = (float*) malloc (size_in);
	h_filter = (float*) malloc (size_filter);
	h_out = (float*) malloc (size_out);

	cudaMallocManaged(&d_in, size_in);
	cudaMallocManaged(&d_filter, size_filter);
	cudaMallocManaged(&d_out, size_out);

	init(d_in, size_in, 1.0);
	init(d_filter, size_filter, 1.0);
	init(d_out, size_out, 0.0);

	size_t num_threads = 256;
	size_t num_blocks = (size_out-1)/num_threads + 1;

	gpu_conv1d<<<num_blocks, num_threads>>>(d_out, d_in, d_filter, size_in, size_filter);
	cudaDeviceSynchronize();

	init(h_in, size_in, 1.0);
	init(h_filter, size_filter, 1.0);
	init(h_out, size_out, 0.0);

	host_conv1d(h_out, h_in, h_filter, size_in, size_filter);
    for (int i =0; i<size_out; i++)
            if (d_out[i] != h_out[i]){
                    printf("Program failed!! Check the idx: %d", i);
                    return 0;
            }
    printf("Success!!\n");
}
