#include "common.h"

int BLOCKSIZE = 256;
bool USING_PARAMETER = false;
bool USING_SHAREDMEM = true;
bool SHAREDMEM_KEY = true;
bool SHAREDMEM_SBOX = true;
bool CONSTANTMEM = false;
bool QUIET = false;
int AES_SIZE = 256;//other options 192, 128
int AES_KEY_EXP_SIZE = 240;
int ABLOCKS_PER_THREAD = 1;

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

bool ingestCommandLineOptions(int aes_size, int blocksize,
		int blocksperthread,
		bool sharedmem_key,
		bool sharedmem_sbox,
		bool parameter,
		bool quiet,
		bool constant) {

	if (aes_size == 256 || aes_size == 192 || aes_size == 128) {
		AES_SIZE = aes_size;
	}
	else {
		printf("ERROR: invalid AES size argument, must be 256, 192, or 128\n");
		return false;
	}
	if (blocksize >= 32 && blocksize % 32 == 0){
	//if (ceil(log2(blocksize)) == floor(log2(blocksize))) {
		BLOCKSIZE = blocksize;
	}
	else {
		printf("ERROR: invalid block size, must be multiple of 32\n");
		return false;
	}
	if (blocksperthread < 1 || blocksperthread > 1024) {
		printf("ERROR: invalid blocksperthread argument, must be between 1 and 1024\n");
		return false;
	}
	ABLOCKS_PER_THREAD	= blocksperthread;
	SHAREDMEM_KEY		= sharedmem_key;
	SHAREDMEM_SBOX		= sharedmem_sbox;
	USING_SHAREDMEM		= SHAREDMEM_KEY | SHAREDMEM_SBOX;
	USING_PARAMETER		= parameter;
	CONSTANTMEM			= constant;
	QUIET				= quiet;

	switch (AES_SIZE) {
	case 256:
		AES_KEY_EXP_SIZE = 240;
		break;
	case 192:
		AES_KEY_EXP_SIZE = 208;
		break;
	case 128:
		AES_KEY_EXP_SIZE = 176;
		break;
	default:
		break;
	}

	if (!QUIET) {
		printf("block %d, key %d, shared %d, skey %d, sbox %d, parameter %d\n",
			BLOCKSIZE, AES_SIZE, USING_SHAREDMEM, SHAREDMEM_KEY, SHAREDMEM_SBOX, USING_PARAMETER);
	}

	return true;
}//ingestCommandLineOptions


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
