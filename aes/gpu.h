#pragma once

#include "common.h"

namespace AES {
    namespace GPU {
        AES::Common::PerformanceTimer& timer();

		long encryptECB(const uint8_t* key, const uint8_t* input, uint8_t* output, uint64_t bufferLength);
		long decryptECB(const uint8_t* key, const uint8_t* input, uint8_t* output, uint64_t bufferLength);
		long encryptCTR(const uint8_t* key, const uint8_t* iv, const uint8_t* input, uint8_t* output, uint64_t bufferLength);
		long decryptCTR(const uint8_t* key, const uint8_t* iv, const uint8_t* input, uint8_t* output, uint64_t bufferLength);

        void scan(int n, int *odata, const int *idata);
    }
}
