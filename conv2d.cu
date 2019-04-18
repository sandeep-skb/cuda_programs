// Assumption is input size == output size


#include <stdio.h>
#include <stdlib.h>


#define mask_width  2
#define block_size  o_tile_width + mask_width - 1
#define o_tile_width 2


__global__
void gpu_conv2d(float *d_out, float *d_in, float *d_filter, int height, int width){
    __shared__ float sh_din[block_size][block_size];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y*o_tile_width + ty;
    int col_o = blockIdx.x * o_tile_width + tx;
    int diff = mask_width - 1;
    int row_i = row_o - diff;
    int col_i = col_o - diff;

    if ((row_i >= 0  && row_i < height) &&
    	(col_i >=0 && col_i < width )){
    	sh_din[ty][tx] = d_in[row_i*width + col_i];
    }
    else{
    	sh_din[ty][tx] = 0.0;
    }
    __syncthreads();

    float output = 0.0;

    if (tx < o_tile_width && ty < o_tile_width){
    	for (int i=0; i < mask_width; i++){
    		for (int j =0; j < mask_width; j++){
    			output += d_filter[i*mask_width + j] * sh_din[ty+i][tx+j]; 
    		}
    	}
    }
	if (tx < o_tile_width && ty < o_tile_width){
		d_out[row_o*width + col_o] = output;
	}

}


void init(float *arr, int h, int w, float val){
	for (int r=0; r < h; r++){
		for (int c=0; c < w; c++){
			arr[r*w + c] = val;
		}
	}
}



void host_conv2d(float *h_out, float *h_in, float *h_filter, int height, int width){

	for (int r=0; r<height; r++){
		for (int c =0; c < width; c++){
			float output = 0.0;
			int r_i = r - mask_width + 1;
			int c_i = c - mask_width + 1;
			//printf("r_i: %d , c_i: %d\n", r_i, c_i);
			for (int mr=0; mr<mask_width; mr++){
		        for (int mc=0; mc< mask_width; mc++){
	                if ( ((r_i+mr) >= 0 && (r_i + mr) < height) && ((c_i+mc) >=0 && (c_i+mc) < width) )
	                    output+= h_in[(mr + r_i)*width + (c_i+mc)] * h_filter[mr*mask_width + mc];
		        }
			}
			h_out[r*width + c] = output;
		}
	}

}



int main(){
	float *d_in, *d_filter, *d_out;
	float *h_in, *h_filter, *h_out; // Only for checking. Not needed for functionality

	int height = 6;
	int width = 6;

	size_t size_in = height*width*sizeof(float);
	size_t size_filter = mask_width*mask_width*sizeof(float);
	size_t size_out = height*width*sizeof(float);

	h_in = (float*) malloc (size_in);
	h_filter = (float*) malloc (size_filter);
	h_out = (float*) malloc (size_out);

	cudaMallocManaged(&d_in, size_in);
	cudaMallocManaged(&d_filter, size_filter);
	cudaMallocManaged(&d_out, size_out);

	init(d_in, height, width, 1.0);
	init(d_filter, mask_width, mask_width, 1.0);
	init(d_out, height, width, 0.0);

	dim3 num_threads (block_size, block_size);
	dim3 num_blocks ((height-1)/(o_tile_width) + 1, (width-1)/(o_tile_width) + 1) ;

	gpu_conv2d<<<num_blocks, num_threads>>>(d_out, d_in, d_filter, height, width);
	cudaDeviceSynchronize();


	init(h_in, height, width, 1.0);
	init(h_filter, mask_width, mask_width, 1.0);
	init(h_out, height, width, 0.0);

	host_conv2d(h_out, h_in, h_filter, height, width);
    
    for(int i=0; i<height; i++){
        for (int j=0; j<width; j++){
                if (d_out[i*width +j] != h_out[i*width +j]){
                        printf(" h_out[%d][%d]: %f", i, j, h_out[i*width + j]);
                        printf(" d_out[%d][%d]: %f", i, j, d_out[i*width + j]);
                        return 0;
                }
        }
    }

	/*
    for (int i =0; i<dout_size; i++)
            if (d_out[i] != h_out[i]){
                    printf("Program failed!! Check the idx: %d", i);
                    return 0;
            }
    */
    printf("Success!!\n");
}
