#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": line %d: %s: %s\n", line, msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace AES {
    namespace Common {
		uint64_t padData(uint8_t* data, uint64_t currentLength) {
			uint64_t newLen = ((currentLength / AES_BLOCKLEN) + 1) * AES_BLOCKLEN;
			uint8_t lenDiff = (uint8_t)(newLen - currentLength);
			for (uint8_t i = 0; i < lenDiff; i++) {
				data[currentLength + i] = lenDiff;
			}

			return newLen;
		}//padData

		uint64_t unpadData(uint8_t* data, uint64_t currentLength) {
#if ASSERTING
			assert(currentLength % AES_BLOCKLEN == 0);
#endif
			uint8_t padAmount = data[currentLength - 1];
#if ASSERTING
			assert(padAmount <= AES_BLOCKLEN);
			assert(padAmount > 0);

			for (uint8_t i = 0; i < padAmount; i++) {
				assert(data[currentLength - 1 - i] == padAmount);
			}
#endif

			uint64_t newLen = currentLength - padAmount;
			return newLen;
		}


    }
}
