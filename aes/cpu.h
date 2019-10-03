#pragma once

#include "common.h"


namespace AES {
    namespace CPU {
	    AES::Common::PerformanceTimer& timer();

		/**
		For all the encrypt functions, assumes the size of the output buffer is already
		equal to the size of the input buffer plus AES_BLOCKLEN
		Returns the size of the new buffer (both encrypted and decrypted)
		*/
		long encryptECB(const uint8_t* key, const uint8_t* input, uint8_t* output, uint64_t bufferLength);
		long decryptECB(const uint8_t* key, const uint8_t* input, uint8_t* output, uint64_t bufferLength);
		long encryptCTR(const uint8_t* key, const uint8_t* iv, const uint8_t* input, uint8_t* output, uint64_t bufferLength);
		long decryptCTR(const uint8_t* key, const uint8_t* iv, const uint8_t* input, uint8_t* output, uint64_t bufferLength);
    }
}
