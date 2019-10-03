#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace AES {
    namespace CPU {
        using AES::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

		long encryptECB(const uint8_t* key, const uint8_t* input, uint8_t* output, uint64_t bufferLength) {
			timer().startCpuTimer();
			//create context/key
			struct AES_ctx context;
			AES_init_ctx(&context, key);

			//encryption happens in place, so start by moving input to output
			memcpy(output, input, bufferLength);
			long inputsize = (long)AES::Common::padData(output, bufferLength);

			//encrypt
			for (long i = 0; i < inputsize / AES_BLOCKLEN; i++) {
				AES_ECB_encrypt(&context, output + i * AES_BLOCKLEN);
			}
			timer().endCpuTimer();

			return inputsize;

		}//encryptECB

		long decryptECB(const uint8_t* key, const uint8_t* input, uint8_t* output, uint64_t bufferLength) {
			timer().startCpuTimer();
			//create context/key
			struct AES_ctx context;
			AES_init_ctx(&context, key);

#if ASSERTING
			assert(bufferLength % AES_BLOCKLEN == 0);
#endif

			//decryption happens in place, so start by moving input to output
			memcpy(output, input, bufferLength);

			
			//decrypt
			for (long i = 0; i < bufferLength / AES_BLOCKLEN; i++) {
				AES_ECB_decrypt(&context, output + i * AES_BLOCKLEN);
			}//for

			//unpad data
			long outputLength = (long)AES::Common::unpadData(output, bufferLength);

			timer().endCpuTimer();

			return outputLength;

		}//decryptECB

		long encryptCTR(const uint8_t* key, const uint8_t* iv, const uint8_t* input, uint8_t* output, uint64_t bufferLength) {
			timer().startCpuTimer();
			//create context/key
			struct AES_ctx context;
			AES_init_ctx_iv(&context, key, iv);

			//encryption happens in place, so start by moving input to output
			memcpy(output, input, bufferLength);
			long inputsize = (long)AES::Common::padData(output, bufferLength);

			//encrypt
			AES_CTR_xcrypt_buffer(&context, output, inputsize);


			timer().endCpuTimer();
			return inputsize;
		}//encryptCTR

		long decryptCTR(const uint8_t* key, const uint8_t* iv, const uint8_t* input, uint8_t* output, uint64_t bufferLength) {
			timer().startCpuTimer();
			//create context/key
			struct AES_ctx context;
			AES_init_ctx_iv(&context, key, iv);

#if ASSERTING
			assert(bufferLength % AES_BLOCKLEN == 0);
#endif

			//decryption happens in place, so start by moving input to output
			memcpy(output, input, bufferLength);

			//decrypt
			AES_CTR_xcrypt_buffer(&context, output, (uint32_t)bufferLength);

			//unpad data
			long outputLength = (long)AES::Common::unpadData(output, bufferLength);

			timer().endCpuTimer();

			return outputLength;

		}//decryptCTR


    }
}
