#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "common.h"
#include "naive.h"
#include <math.h>

namespace AES {
    namespace Naive {
        using AES::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		//#####################
		// FUNCTION DEFINITIONS
		//#####################
		
		//__device__ void kShiftIncEx(int N, int index, int* idata, int* odata);
		//__device__ void scanStep(int N, int index, unsigned long stepLevel, int* idata, int* odata);
		//__device__ void kmoveData(int N, int index, int* odata, int* idata);
		//__global__ void kScan(int N, int* idata, int* odata, int numLevels);

		//####################
		// LITTLE HELPERS
		//####################

		__host__ __device__ uint64_t getPaddedLength(uint64_t currentLength) {
			uint64_t newLen = ((currentLength / AES_BLOCKLEN) + 1) * AES_BLOCKLEN;
			return newLen;
		}

		//####################
		// GLOBAL MEMORY
		//####################

		uint8_t* d_input;
		uint8_t* d_output;


		int* kern_idata;
		int* kern_odata;

		//####################
		// KERNEL FUNCTIONS
		//####################

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

		//######################
		// ENCRYPTION MAIN FUNCS
		//######################

		long encryptECB(const uint8_t* key, const uint8_t* input, uint8_t* output, uint64_t bufferLength) {
			uint64_t paddedLength = getPaddedLength(bufferLength);
			uint8_t lenDiffArray[AES_BLOCKLEN] = {};
			uint8_t lenDiff = (uint8_t)(paddedLength - bufferLength);
			for (uint8_t i = 0; i < AES_BLOCKLEN; i++) {
				lenDiffArray[i] = lenDiff;
			}

			//malloc space for padded input/output
			cudaMalloc((void**)&d_input, paddedLength * sizeof(uint8_t));
			cudaMalloc((void**)&d_output, paddedLength * sizeof(uint8_t));
			checkCUDAError("CudaMalloc");
			//copy input
			cudaMemcpy(d_input, input, bufferLength * sizeof(uint8_t), cudaMemcpyHostToDevice);
			checkCUDAError("CudaMemcpy");
			//pad input
			cudaMemcpy(d_input + bufferLength, lenDiffArray, lenDiff, cudaMemcpyHostToDevice);
			checkCUDAError("CudaMemcpy");

			timer().startGpuTimer();

			//TODO: actually encrypt



			timer().endGpuTimer();

			//copy output
			cudaMemcpy(output, d_output, paddedLength * sizeof(uint8_t), cudaMemcpyDeviceToHost);
			checkCUDAError("CudaMemcpy");

			//free input/output
			cudaFree((void*)d_input);
			cudaFree((void*)d_output);
			checkCUDAError("CudaFree");
			return (long)paddedLength;
		}

		long decryptECB(const uint8_t* key, const uint8_t* input, uint8_t* output, uint64_t bufferLength) {
			return -1;
		}

		long encryptCTR(const uint8_t* key, const uint8_t* iv, const uint8_t* input, uint8_t* output, uint64_t bufferLength) {
			uint64_t paddedLength = getPaddedLength(bufferLength);
			uint8_t lenDiffArray[AES_BLOCKLEN] = {};
			uint8_t lenDiff = (uint8_t)(paddedLength - bufferLength);
			for (uint8_t i = 0; i < AES_BLOCKLEN; i++) {
				lenDiffArray[i] = lenDiff;
			}

			//malloc space for padded input/output
			cudaMalloc((void**)&d_input, paddedLength * sizeof(uint8_t));
			cudaMalloc((void**)&d_output, paddedLength * sizeof(uint8_t));
			checkCUDAError("CudaMalloc");
			//copy input
			cudaMemcpy(d_input, input, bufferLength * sizeof(uint8_t), cudaMemcpyHostToDevice);
			checkCUDAError("CudaMemcpy");
			//pad input
			cudaMemcpy(d_input + bufferLength, lenDiffArray, lenDiff, cudaMemcpyHostToDevice);
			checkCUDAError("CudaMemcpy");

			timer().startGpuTimer();

			//TODO: actually encrypt



			timer().endGpuTimer();

			//copy output
			cudaMemcpy(output, d_output, paddedLength * sizeof(uint8_t), cudaMemcpyDeviceToHost);
			checkCUDAError("CudaMemcpy");

			//free input/output
			cudaFree((void*)d_input);
			cudaFree((void*)d_output);
			checkCUDAError("CudaFree");
			return (long)paddedLength;
		}

		long decryptCTR(const uint8_t* key, const uint8_t* iv, const uint8_t* input, uint8_t* output, uint64_t bufferLength) {
			return -1;
		}
    }
}
