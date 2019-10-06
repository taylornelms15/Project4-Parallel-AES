#pragma once

#include "common.h"

/**
Stores the "state" during encryption that gets modified
IMPORTANT NOTE: the input is consumed in these column-wise
Specifically, we index as [colNumber][rowNumber]
*/
typedef struct State {
#if USING_VECTORS
	uchar4 data[4];
#else
	uint8_t data[4][4];
#endif
} State;


namespace AES {
    namespace GPU {
        AES::Common::PerformanceTimer& timer();

		long encryptECB(const uint8_t* key, const uint8_t* input, uint8_t* output, uint64_t bufferLength);
		long decryptECB(const uint8_t* key, const uint8_t* input, uint8_t* output, uint64_t bufferLength);
		long encryptCTR(const uint8_t* key, const uint8_t* iv, const uint8_t* input, uint8_t* output, uint64_t bufferLength);
		long decryptCTR(const uint8_t* key, const uint8_t* iv, const uint8_t* input, uint8_t* output, uint64_t bufferLength);

		/**
		Expands the key to make a round schedule
		Note: this is lifted directly from the library implementation, as it is more efficient to run on the CPU than the GPU (given the data size)
		As such, a rigorous re-implementation is out of the scope of this project
		*/
		void expandKey(const uint8_t* key, uint8_t* expanded, uint8_t Nk, uint8_t Nr);
    }
}
