#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		//###########################
		// MEMORY POINTERS
		//###########################
		int* kern_idata;
		int* kern_odata;
		int* kern_tdata;


		//###########################
		// UPSWEEP
		//###########################

		/**
		* Does the actual data movement for the upsweep
		* Organized a bit strange to make sure threads doing work are centered around the lower warps, rather than spread out
		* Not sure whether or not is more efficient
		*/
		__global__ void upsweepMove(long N, int currentLevel, int* idata) {
			long threadnum = threadIdx.x + (blockIdx.x * blockDim.x);
			long multiplier = 1 << currentLevel;
			unsigned long index = (multiplier * (threadnum + 1)) - 1;
			if (index >= N) return;

			multiplier = multiplier >> 1;//turns into the gap we're facing

			idata[index] = idata[index] + idata[index - multiplier];

		}//uspweepMove


		void upsweep(long N, int numLevels, int* idata, dim3 tpb, dim3 bpg) {

			int currentLevel = 1;
			while (currentLevel <= numLevels) {

				upsweepMove<<<bpg, tpb>>>(N, currentLevel, idata);

				currentLevel++;
				cudaDeviceSynchronize();
			}//while

		}//upsweep

		//###########################
		// DOWNSWEEP
		//###########################

		__global__ void downsweepStep(int N, int levelWidth, int* idata) {
			long index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (!((index + 1) % levelWidth)) {
				int jumpDist = levelWidth / 2;
				int temp = idata[index];
				idata[index] = idata[index] + idata[index - jumpDist];
				idata[index - jumpDist] = temp;
			}//if we're on a node this level
		}//downsweepStep


		void downsweep(int N, int numLevels, int* idata, dim3 tpb, dim3 bpg){

			//this variable does that first "set root to 0" step
			int fakeZero = 0;
			cudaMemcpy(idata + N - 1, &fakeZero, sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("CudaMemcpy failed!\n", NULL, __LINE__);
			
			int currentLevel = 0;
			int levelWidth;
			while (currentLevel < numLevels) {
				levelWidth = N >> currentLevel;

				downsweepStep<<<bpg, tpb>>>(N, levelWidth, idata);

				cudaDeviceSynchronize();

				currentLevel++;
			}//while

		}//downsweep

		//###########################
		// COMPACTION
		//###########################

		__global__ void makeTempArray(int n, int N, int* idata, int* tdata) {
			long index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) return;
			if (index >= n) {
				tdata[index] = 0;
			}//if
			else {
				if (idata[index]) tdata[index] = 1;
				else tdata[index] = 0;
			}//else
		}//makeTempArray

		__global__ void scatter(int n, int* idata, int* tdata, int* odata) {
			long index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) return;

			if (idata[index]) {
				odata[tdata[index]] = idata[index];
			}//if we're putting this into the result
			
		}//scatter

		
		//###########################
		// CPU (MAIN) FUNCTIONS
		//###########################


		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int* odata, const int* idata) {
			int numLevels = ilog2ceil(n);
			int N = 1 << numLevels;//pad out to this many elements
			int numToFake = N - n;
			int* fakeZeroes = (int*)malloc(numToFake * sizeof(int));
			for (int i = 0; i < numToFake; i++) fakeZeroes[i] = 0;

			int blocksPerGrid = (N + BLOCKSIZE - 1) / BLOCKSIZE;
			dim3 bpg = dim3(blocksPerGrid);
			dim3 tpb = dim3(BLOCKSIZE);


			cudaMalloc((void**)& kern_idata, N * sizeof(int));
			checkCUDAErrorFn("cudaMalloc kern_idata failed!\n", NULL, __LINE__);

			cudaMemcpy(kern_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy kern_idata failed!\n", NULL, __LINE__);
			//pad with zeroes
			cudaMemcpy(&(kern_idata[n]), fakeZeroes, numToFake * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy failed!\n", NULL, __LINE__);

			timer().startGpuTimer();

			//Upsweep on kern_idata
			upsweep(N, numLevels, kern_idata, tpb, bpg);
			checkCUDAErrorFn("upsweep failed!\n", NULL, __LINE__);

			//Downsweep on kern_idata

			//actual downsweep
			downsweep(N, numLevels, kern_idata, tpb, bpg);
			checkCUDAErrorFn("downsweep failed!\n", NULL, __LINE__);

			timer().endGpuTimer();

			cudaMemcpy(odata, kern_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy kern_idata failed!\n", NULL, __LINE__);

			cudaFree(kern_idata);
			checkCUDAErrorFn("cudaFree failed!\n", NULL, __LINE__);
			free(fakeZeroes);
		}//doscan

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

			int numLevels = ilog2ceil(n);
			int N = 1 << numLevels;//pad out to this many elements
			int numToFake = N - n;

			dim3 tpb = dim3(BLOCKSIZE);
			dim3 bpg = dim3((n + BLOCKSIZE - 1) / BLOCKSIZE);
			dim3 tpbN = dim3(BLOCKSIZE);
			dim3 bpgN = dim3((N + BLOCKSIZE - 1) / BLOCKSIZE);



			cudaMalloc((void**)& kern_idata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc kern_idata failed!\n", NULL, __LINE__);
			cudaMalloc((void**)& kern_tdata, N * sizeof(int));
			checkCUDAErrorFn("cudaMalloc kern_odata failed!\n", NULL, __LINE__);
			cudaMalloc((void**)& kern_odata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc kern_odata failed!\n", NULL, __LINE__);

			cudaMemcpy(kern_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy kern_idata failed!\n", NULL, __LINE__);

			timer().startGpuTimer();

			//make our temporary binary array
			makeTempArray<<<bpgN, tpbN>>>(n, N, kern_idata, kern_tdata);
			checkCUDAErrorFn("tempArray failed!\n", NULL, __LINE__);
			cudaDeviceSynchronize();

			//Scan the binary array

			//Upsweep on kern_idata
			upsweep(N, numLevels, kern_tdata, tpbN, bpgN);
			checkCUDAErrorFn("upsweep failed!\n", NULL, __LINE__);

			//Downsweep on kern_idata
			downsweep(N, numLevels, kern_tdata, tpbN, bpgN);
			checkCUDAErrorFn("downsweep failed!\n", NULL, __LINE__);

			//scatter
			//get the ending size of the odata
			int outputSize = -1;
			cudaMemcpy(&outputSize, kern_tdata + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			if (idata[n - 1]) outputSize++;//necessary because tdata holds the exclusive scan, not inclusive


			//actually scatter
			scatter<<<bpg, tpb>>>(n, kern_idata, kern_tdata, kern_odata);

			timer().endGpuTimer();

			cudaMemcpy(odata, kern_odata, outputSize * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy failed!\n", NULL, __LINE__);

            
			cudaFree(kern_idata);
			checkCUDAErrorFn("cudaFree failed!\n", NULL, __LINE__);
			cudaFree(kern_tdata);
			checkCUDAErrorFn("cudaFree failed!\n", NULL, __LINE__);
			cudaFree(kern_odata);
			checkCUDAErrorFn("cudaFree failed!\n", NULL, __LINE__);


			return outputSize;
        }
    }
}
