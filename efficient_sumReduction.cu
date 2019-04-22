// This implementation uses consecutive threads. This reduces the control divergence as all the threads in the warps are active
// as oppose to the previous implementation.

#include<stdio.h>                         
#include<stdlib.h>                        
#define block_size 128                    


__global__
void gpu_reduceSum(float *d_out, float *d_in, int din_size){
    int num_elements = 2*block_size;                    
    __shared__ float shmem[2*block_size];               
    int start = blockIdx.x*num_elements;                
    int tx = threadIdx.x;                               
    if (start + tx < din_size)                          
        shmem[tx] = d_in[start + tx];               
    if (start + tx + block_size < din_size)             
        shmem[block_size + tx] = d_in[start + block_size + tx];

    for(int stride=blockDim.x; stride > 0; stride /= 2){

        __syncthreads();
        if(tx < stride)
            shmem[tx] += shmem[tx + stride];
    }                                                   

    d_out[blockIdx.x] = shmem[0];
}                                    


void init(float *arr, int N, float val){
    for (int r=0; r < N; r++){      
        arr[r] = val;   
    }                               
}                                       

void host_reduceSum(float *h_out, float *h_in, int din_size){
    for (int i =0; i < din_size; i++){                   
        *h_out += h_in[i];                           
    }                                                    
}                                                            

int main(){
    float *d_in, *d_out;
    float *h_in, *h_out; // Only for checking. Not needed for functionality


    int din_size = 1024;
    int elements = 2*block_size;
    int dout_size = (din_size-1)/elements + 1;

    //calculate memory
    size_t size_in = din_size*sizeof(float);
    size_t size_out = dout_size*sizeof(float);

    //allocate memory in host
    h_in = (float*) malloc (size_in);
    h_out = (float*) malloc (1*sizeof(float));

    //allocate memory using UVM
    cudaMallocManaged(&d_in, size_in);
    cudaMallocManaged(&d_out, size_out);

//Initialize device parameters
    init(d_in, din_size, 1.0);
    init(d_out, dout_size, 0.0);

    int num_threads = block_size;
    int num_blocks = (din_size-1)/elements + 1;

    //gpu sum reduction
    //printf("num_blocks: %d, num_threads: %d\n", num_blocks, num_threads);
    gpu_reduceSum<<<num_blocks, num_threads>>>(d_out, d_in, din_size);
    cudaDeviceSynchronize();

    init(h_in, din_size, 1.0);
    init(h_out, dout_size, 0.0);

    host_reduceSum(h_out, h_in, din_size);

    float final_out = 0.0;
    for (int i=0; i<num_blocks; i++){
        final_out += d_out[i];
    }
    //printf("h_out: %f\n", *h_out);
    if (final_out != *h_out){
        printf("Failure!!\n");
        return 0;
    }
    printf("Success!!\n");
}
