#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <sys/time.h>
extern "C" {
    #include "libs/bitmap.h"
}

#define ERROR_EXIT -1

/* Divide the problem into blocks of BLOCKX x BLOCKY threads */
#define BLOCKY 32
#define BLOCKX 32


/* Problem size */
//#define XSIZE 2560
//#define YSIZE 2048
#define FILTERDIM 3

// Generalize PIXEL(i,j) to support every horizontal res
#define PIXEL(i,j,width) ((i)+(j)*(width))


#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Convolutional Filter Examples, each with dimension 3,
// gaussian filter with dimension 5
// If you apply another filter, remember not only to exchange
// the filter but also the filterFactor and the correct dimension.

int const laplacian1Filter[] = {  -1,  -4,  -1,
                                 -4,  20,  -4,
                                 -1,  -4,  -1};

float const laplacian1FilterFactor = (float) 1.0;

/*
int const sobelYFilter[] = {-1, -2, -1,
                             0,  0,  0,
                             1,  2,  1};
float const sobelYFilterFactor = (float) 1.0;

int const sobelXFilter[] = {-1, -0, -1,
                            -2,  0, -2,
                            -1,  0, -1 , 0};
float const sobelXFilterFactor = (float) 1.0;

int const laplacian2Filter[] = { 0,  1,  0,
                                 1, -4,  1,
                                 0,  1,  0};
float const laplacian2FilterFactor = (float) 1.0;

int const laplacian3Filter[] = { -1,  -1,  -1,
                                  -1,   8,  -1,
                                  -1,  -1,  -1};
float const laplacian3FilterFactor = (float) 1.0;


//Bonus Filter:

int const gaussianFilter[] = { 1,  4,  6,  4, 1,
                               4, 16, 24, 16, 4,
                               6, 24, 36, 24, 6,
                               4, 16, 24, 16, 4,
                               1,  4,  6,  4, 1 };
x
float const gaussianFilterFactor = (float) 1.0 / 256.0;
*/

// Apply convolutional filter on image data (CPU - serial version)
void host_applyFilter(unsigned char **out, unsigned char **in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
  unsigned int const filterCenter = (filterDim / 2);
  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = 0; x < width; x++) {
      int aggregate = 0;
      for (unsigned int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++) {
          int nkx = filterDim - 1 - kx;

          int yy = y + (ky - filterCenter);
          int xx = x + (kx - filterCenter);
          if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
            aggregate += in[yy][xx] * filter[nky * filterDim + nkx];
        }
      }
      aggregate *= filterFactor;
      if (aggregate > 0) {
        out[y][x] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[y][x] = 0;
      }
    }
  }
}

// Apply convolutional filter on image data (GPU - basic device kernel function)
__global__ void device_applyFilter(unsigned char *out, unsigned char *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
	unsigned int const filterCenter = (filterDim / 2);
	unsigned int y = blockIdx.y * BLOCKY + threadIdx.y;
	unsigned int x = blockIdx.x * BLOCKX + threadIdx.x;
	
	int aggregate = 0;
	for (unsigned int ky = 0; ky < filterDim; ky++) {
		int nky = filterDim - 1 - ky;
		for (unsigned int kx = 0; kx < filterDim; kx++) {
			int nkx = filterDim - 1 - kx;

			int yy = y + (ky - filterCenter);
			int xx = x + (kx - filterCenter);
			if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
				aggregate += in[PIXEL(xx, yy, width)] * filter[nky * filterDim + nkx];
		}
	}
	aggregate *= filterFactor;
	if (aggregate > 0) {
		out[PIXEL(x, y, width)] = (aggregate > 255) ? 255 : aggregate;
	}
	else {
		out[PIXEL(x, y, width)]= 0;
	}
}

// Apply convolutional filter on image data (GPU - device kernel function using shared memory)
__global__ void s_device_applyFilter(unsigned char *out, unsigned char *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
	unsigned int const filterCenter = (filterDim / 2);
	unsigned int y = blockIdx.y * BLOCKY + threadIdx.y;
	unsigned int x = blockIdx.x * BLOCKX + threadIdx.x;
	

	__shared__ int s_filter[FILTERDIM * FILTERDIM];
	if (threadIdx.x < filterDim && threadIdx.y < filterDim) {
			s_filter[PIXEL(threadIdx.x, threadIdx.y, filterDim)] = filter[PIXEL(threadIdx.x, threadIdx.y, filterDim)];
	}

	//__shared__ unsigned char s_in[BLOCKX * BLOCKY];
	//s_in[PIXEL(threadIdx.x, threadIdx.y , BLOCKX)] = in[PIXEL(x, y, width)];


	int aggregate = 0;
	__syncthreads();
	for (unsigned int ky = 0; ky < filterDim; ky++) {
		int nky = filterDim - 1 - ky;
		int dy = ky - filterCenter;
		int yy = y + dy;
		for (unsigned int kx = 0; kx < filterDim; kx++) {
			int nkx = filterDim - 1 - kx;
			int dx = kx - filterCenter;
			//if (threadIdx.x >= filterCenter && threadIdx.x < BLOCKX - filterCenter && threadIdx.y >= filterCenter && threadIdx.y < BLOCKY - filterCenter) {
			//	aggregate += s_in[PIXEL(threadIdx.x + dx, threadIdx.y + dy , BLOCKX)] * s_filter[nky * filterDim + nkx];
			//	continue;
			//}

			int xx = x + dx;
			if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height) {
				aggregate += in[PIXEL(xx, yy, width)] * s_filter[nky * filterDim + nkx];
			}
		}
	}
	aggregate *= filterFactor;
	if (aggregate > 0) {
		out[PIXEL(x, y, width)] = (aggregate > 255) ? 255 : aggregate;
	}
	else {
		out[PIXEL(x, y, width)]= 0;
	}
}

void help(char const *exec, char const opt, char const *optarg) {
    FILE *out = stdout;
    if (opt != 0) {
        out = stderr;
        if (optarg) {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        } else {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s in.bmp out.bmp -i 10000\n", exec);
}

double walltime ( void ) {
	static struct timeval t;
	gettimeofday ( &t, NULL );
	return ( t.tv_sec + 1e-6 * t.tv_usec );
}

int main(int argc, char **argv) {
	/*
	Parameter parsing, don't change this!
	*/
	unsigned int iterations = 1;
	char *output = NULL;
	char *input = NULL;
	int ret = 0;

	static struct option const long_options[] =  {
		{"help",       no_argument,       0, 'h'},
		{"iterations", required_argument, 0, 'i'},
		{0, 0, 0, 0}
	};

	static char const * short_options = "hi:";

	{
	char *endptr;
	int c;
	int option_index = 0;
	while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
		switch (c) {
		case 'h':
			help(argv[0],0, NULL);
			return 0;
		case 'i':
			iterations = strtol(optarg, &endptr, 10);
			if (endptr == optarg) {
			  help(argv[0], c, optarg);
			  return ERROR_EXIT;
			}
		break;
		default:
			abort();
	  }
	}
	}

	if (argc <= (optind+1)) {
		help(argv[0],' ',"Not enough arugments");
		return ERROR_EXIT;
	}
	input = (char *)calloc(strlen(argv[optind]) + 1, sizeof(char));
	strncpy(input, argv[optind], strlen(argv[optind]));
	optind++;

	output = (char *)calloc(strlen(argv[optind]) + 1, sizeof(char));
	strncpy(output, argv[optind], strlen(argv[optind]));
	optind++;

	/*
	End of Parameter parsing!
	*/

	/*
	Create the BMP image and load it from disk.
	*/
	bmpImage *image = newBmpImage(0,0);
	if (image == NULL) {
		fprintf(stderr, "Could not allocate new image!\n");
	}

	if (loadBmpImage(image, input) != 0) {
		fprintf(stderr, "Could not load bmp image '%s'!\n", input);
		freeBmpImage(image);
		return ERROR_EXIT;
	}


	// Create a single color channel image. It is easier to work just with one color
	bmpImageChannel *imageChannel = newBmpImageChannel(image->width, image->height);
	if (imageChannel == NULL) {
		fprintf(stderr, "Could not allocate new image channel!\n");
		freeBmpImage(image);
		return ERROR_EXIT;
	}

	// Extract from the loaded image an average over all colors - nothing else than
	// a black and white representation
	// extractImageChannel and mapImageChannel need the images to be in the exact
	// same dimensions!
	// Other prepared extraction functions are extractRed, extractGreen, extractBlue
	if(extractImageChannel(imageChannel, image, extractAverage) != 0) {
		fprintf(stderr, "Could not extract image channel!\n");
		freeBmpImage(image);
		freeBmpImageChannel(imageChannel);
		return ERROR_EXIT;
	}
	int kerDim = FILTERDIM; // Dimension of currently used kernel

	// Assign image dimensions onto variables
	int im_XSIZE = imageChannel->width;
	int im_YSIZE = imageChannel->height;

	// Variables for used for wall-times.
	double start;
	double hosttime=0;
	double devicetime=0;

	/*																					*/
	/*									CPU PROCESSING									*/
	/*																					*/

	// Intitialize host memory for CPU processing.
	// imageChannel->data is a 2-dimensional array of unsigned char which is accessed row first ([y][x]).
	bmpImageChannel *processImageChannel = newBmpImageChannel(im_XSIZE, im_YSIZE);

	//Here we do the actual computation!
	// Host computation
	start = walltime();
	for (unsigned int i = 0; i < 0; i ++) {
		host_applyFilter(processImageChannel->data,
				imageChannel->data,
				im_XSIZE,
				im_YSIZE,
				(int *)laplacian1Filter, kerDim, laplacian1FilterFactor
	//               (int *)laplacian2Filter, 3, laplacian2FilterFactor
	//               (int *)laplacian3Filter, 3, laplacian3FilterFactor
	//               (int *)gaussianFilter, 5, gaussianFilterFactor
				);
		//Swap the data pointers
		unsigned char ** tmp1 = processImageChannel->data;
		processImageChannel->data = imageChannel->data;
		imageChannel->data = tmp1;
		unsigned char * tmp_raw = processImageChannel->rawdata;
		processImageChannel->rawdata = imageChannel->rawdata;
		imageChannel->rawdata = tmp_raw;
	}
	hosttime = walltime() - start;

	// Free host memory
	freeBmpImageChannel(processImageChannel);

	/*																					*/
	/*									GPU PROCESSING									*/
	/*																					*/
	// Initialize device memory for GPU processing.
	unsigned char *devChannel; // Resultant array
	unsigned char *devProcChannel; // Imageprocessing array
	int *devKernel;	// Filter array (Laplacian1 filter)

	cudaErrorCheck(cudaMalloc((void **) &devChannel, im_XSIZE * im_YSIZE));
	cudaErrorCheck(cudaMalloc((void **) &devProcChannel, im_XSIZE * im_YSIZE));
	cudaErrorCheck(cudaMalloc((void **) &devKernel, kerDim * kerDim * sizeof(int)));

	// Copy the memory data from original host imageChannel to device memory.
	cudaErrorCheck(cudaMemcpy(devChannel, imageChannel->rawdata, im_XSIZE * im_YSIZE, cudaMemcpyHostToDevice));

	// Copy the kernel data from host to device memory.
	cudaErrorCheck(cudaMemcpy(devKernel, laplacian1Filter, kerDim * kerDim * sizeof(int), cudaMemcpyHostToDevice));

	dim3 gridBlock(im_XSIZE / BLOCKX, im_YSIZE / BLOCKY);
	dim3 threadBlock(BLOCKX, BLOCKY);

	// Device computation
	start = walltime();
	for (unsigned int i = 0; i < iterations; i ++) {
		s_device_applyFilter<<<gridBlock, threadBlock>>>(devProcChannel,
					devChannel,
					im_XSIZE,
					im_YSIZE,
					devKernel, kerDim, laplacian1FilterFactor
					);
		cudaErrorCheck(cudaGetLastError());

		//Swap the data pointers
		unsigned char *tmp2 = devProcChannel;
		devProcChannel = devChannel;
		devChannel = tmp2;
	}
	devicetime = walltime() - start;
	// Free device memory used in processing
	cudaFree(devProcChannel);
	cudaFree(devKernel);

	// Initialize host memory for GPU processed image
	bmpImageChannel *gpuImageChannel = newBmpImageChannel(im_XSIZE, im_YSIZE);

	// Copy processed image memory from device to host
	cudaMemcpy(gpuImageChannel->rawdata, devChannel, im_XSIZE * im_YSIZE, cudaMemcpyDeviceToHost);
	
	cudaFree(devChannel);

	int errors=0;
	/* check if result is correct */
	for(unsigned int k = 0; k<im_XSIZE * im_YSIZE; k++) {
		if (imageChannel->rawdata[k] != gpuImageChannel->rawdata[k]) {
			errors++;
		}
	}
	
	if(errors>0) {
		printf("Found %d errors of the %d pixels.\n",errors, im_XSIZE * im_YSIZE);
	}
	else {
		puts("Device calculations are correct.");
	}

	printf("\n");
	printf("Host time:          %7.5f ms\n",hosttime*1e3);
	printf("Device calculation: %7.5f ms\n",devicetime*1e3);


	// Map our single color image back to a normal BMP image with 3 color channels
	// mapEqual puts the color value on all three channels the same way
	// other mapping functions are mapRed, mapGreen, mapBlue
	if (mapImageChannel(image, gpuImageChannel, mapEqual) != 0) {
		fprintf(stderr, "Could not map image channel!\n");
		freeBmpImage(image);
		freeBmpImageChannel(gpuImageChannel);
		return ERROR_EXIT;
	}
	freeBmpImageChannel(imageChannel);
	freeBmpImageChannel(gpuImageChannel);

	//Write the image back to disk
	if (saveBmpImage(image, output) != 0) {
		fprintf(stderr, "Could not save output to '%s'!\n", output);
		freeBmpImage(image);
		return ERROR_EXIT;
	};

	ret = 0;
	if (input)
		free(input);
	if (output)
		free(output);
	return ret;
};
