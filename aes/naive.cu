#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "common.h"
#include "naive.h"
#include <math.h>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		//####################
		//FUNCTION DEFINITIONS
		//####################
		
		__device__ void kShiftIncEx(int N, int index, int* idata, int* odata);
		__device__ void scanStep(int N, int index, unsigned long stepLevel, int* idata, int* odata);
		__device__ void kmoveData(int N, int index, int* odata, int* idata);
		__global__ void kScan(int N, int* idata, int* odata, int numLevels);

		int* kern_idata;
		int* kern_odata;

		__global__ void scanStep(int N, unsigned long stepLevel, int* idata, int* odata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) return;
			if (index < stepLevel) {
				odata[index] = idata[index];
				return;
			}//if low index
			odata[index] = idata[index] + idata[index - stepLevel];
			return;
		}//scanStep

		/**
		* Note: the numLevels is that log_2(N), and should be computed GPU-side
		*
		__global__ void kScan(int N, int* idata, int* odata, int numLevels) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int currentLevel = 0;
			unsigned long stepLevel = 1;
			while (currentLevel < numLevels) {
				stepLevel = 1 << currentLevel;
				scanStep(N, index, stepLevel, idata, odata);
				__syncthreads();
				kmoveData(N, index, odata, idata);
				__syncthreads();
				currentLevel++;
			}//while

			kShiftIncEx(N, index, idata, odata);

		}//kScan
		*/

		void scanWrapper(int N, int* idata, int* odata, int numLevels) {
			dim3 bpg = dim3((N + BLOCKSIZE - 1) / BLOCKSIZE);
			dim3 tpb = dim3(BLOCKSIZE);
			int currentLevel = 0;
			unsigned long stepLevel = 1;
			while (currentLevel < numLevels) {
				stepLevel = 1 << currentLevel;
				scanStep<<<bpg, tpb>>>(N, stepLevel, idata, odata);
				cudaDeviceSynchronize();
				cudaMemcpy(idata, odata, N * sizeof(int), cudaMemcpyDeviceToDevice);
				currentLevel++;
			}//while

		}//scanWrapper

		/**
		* Shifts our inclusive scan over to be an exclusive scan
		*
		__device__ void kShiftIncEx(int N, int index, int* idata, int* odata) {
			if (index >= N) return;
			else if (index == 0) odata[index] = 0;
			else {
				odata[index] = idata[index - 1];
			}
		}//kShiftIncEx
		*/

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int threadsPerBlock = (n + BLOCKSIZE - 1) / BLOCKSIZE;
			dim3 tpb = dim3(threadsPerBlock);
			dim3 bpg = dim3(BLOCKSIZE);


			//Allocate memory
			cudaMalloc((void**)& kern_idata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc kern_idata failed!\n", NULL, __LINE__);
			cudaMalloc((void**)& kern_odata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc kern_odata failed!\n", NULL, __LINE__);

			//copy input over
			cudaMemcpy(kern_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy kern_idata failed!\n", NULL, __LINE__);

			timer().startGpuTimer();

			int numLevels = ilog2ceil(n);

			scanWrapper(n, kern_idata, kern_odata, numLevels);
			checkCUDAErrorFn("kScan failed!\n", NULL, __LINE__);

			timer().endGpuTimer();

			cudaMemcpy(odata + 1, kern_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy kern_odata failed!\n", NULL, __LINE__);
			//odata[n - 1] = odata[n - 2] + idata[n - 1];
			odata[0] = 0;

			cudaFree(kern_idata);
			cudaFree(kern_odata);

        }
    }
}
