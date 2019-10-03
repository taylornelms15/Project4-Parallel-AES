#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		int* kern_idata;
		int* kern_odata;


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			cudaMalloc((void**)& kern_idata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc kern_idata failed!\n", NULL, __LINE__);
			cudaMalloc((void**)& kern_odata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc kern_odata failed!\n", NULL, __LINE__);
			cudaMemcpy(kern_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy kern_idata failed!\n", NULL, __LINE__);

            timer().startGpuTimer();
			thrust::device_ptr<int> tidata = thrust::device_ptr<int>(kern_idata);
			thrust::device_ptr<int> todata = thrust::device_ptr<int>(kern_odata);
            // TODO use `thrust::exclusive_scan`

			thrust::exclusive_scan(tidata, tidata + n, todata);
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            timer().endGpuTimer();

			cudaMemcpy(odata, kern_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy failed!\n", NULL, __LINE__);
			cudaFree(kern_idata);
			checkCUDAErrorFn("cudaFree failed!\n", NULL, __LINE__);
			cudaFree(kern_odata);
			checkCUDAErrorFn("cudaFree failed!\n", NULL, __LINE__);
        }
    }
}
