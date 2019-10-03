#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int sum = 0;
			for (int i = 0; i < n; i++) {
				odata[i] = sum;
				sum += idata[i];
			}//for
            // TODO
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int numGood = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i]) {
					odata[numGood++] = idata[i];
				}//if
			}//for
	        timer().endCpuTimer();
			return numGood;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int* tempPresence = (int*)malloc(n * sizeof(int));
			int* scannedPresence = (int*)malloc(n * sizeof(int));
			for (int i = 0; i < n; i++) {
				if (idata[i]) {
					tempPresence[i] = 1;
				}//if
				else tempPresence[i] = 0;
			}//for
			
			//scan(n, scannedPresence, tempPresence);

			int sum = 0;
			for (int i = 0; i < n; i++) {
				scannedPresence[i] = sum;
				sum += tempPresence[i];
			}//for

			int numElements = scannedPresence[n - 1];
			for (int i = 0; i < n; i++) {
				if (idata[i]) {
					odata[scannedPresence[i]] = idata[i];
				}//if
			}//for

			free(tempPresence);
			free(scannedPresence);
	        timer().endCpuTimer();
			return  numElements;
        }
    }
}
