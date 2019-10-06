/**
 * @file      main.cpp
 * @brief     AES Test Program
 * @authors   Taylor Nelms
 * @date      2019
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <aes/common.h>
#include <aes/cpu.h>
#include <aes/gpu.h>
#include <cxxopts.hpp>
#include "testing_helpers.hpp"

#define OFFPOT 1

const unsigned int RANDSEED = 0xbad1bad2;
const unsigned int RANDSEED2 = 0x0123fed5;
const unsigned int RANDSEED3 = 0x56781234;

const int SIZE = 1 << 26; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
#if OFFPOT
const int ASIZE = NPOT;
#else
const int ASIZE = SIZE
#endif


int main(int argc, char* argv[]) {
	//parse arguments
	cxxopts::Options options("aes_test", "Program to test parallel AES encryption/decryption");
	options.add_options()
		("k,keysize", "Size of AES key", cxxopts::value<int>()->default_value("256"))
		("b,blocksize", "Size of cuda blocks to use", cxxopts::value<int>()->default_value("256"))
		("n,blocksperthread", "How many AES blocks each thread processes", cxxopts::value<int>()->default_value("1"))
		("y,sharedmemKey", "Whether storing key in shared memory")
		("s,sharedmemSBox", "Whether storing sbox in shared memory")
		("p,parameter", "Whether we are pulling keys and sboxes from a passed parameter")
		("q,quiet", "Runs just the GPU tests, outputs just the timings for automatic consumption")
		;
	auto result = options.parse(argc, argv);
	bool sharedmemKey = false; bool sharedmemSBox = false; bool parameter = false; bool quiet = false;
	int blocksize; int keysize; int blocksperthread;
	sharedmemKey	= result.count("sharedmemKey") > 0;
	sharedmemSBox	= result.count("sharedmemSBox") > 0;
	parameter		= result.count("parameter") > 0;
	quiet			= result.count("quiet") > 0;
	blocksize		= result["blocksize"].as<int>();
	keysize			= result["keysize"].as<int>();
	blocksperthread = result["blocksperthread"].as<int>();

	ingestCommandLineOptions(keysize, blocksize, blocksperthread, sharedmemKey, sharedmemSBox, parameter, quiet);

	//Run tests
	uint8_t* a = (uint8_t*)malloc(ASIZE * sizeof(uint8_t));
	uint8_t* b = (uint8_t*)malloc((ASIZE + AES_BLOCKLEN) * sizeof(uint8_t));
	uint8_t* c = (uint8_t*)malloc(ASIZE * sizeof(uint8_t));
	uint8_t* d = (uint8_t*)malloc((ASIZE + AES_BLOCKLEN) * sizeof(uint8_t));
	uint8_t* e = (uint8_t*)malloc(ASIZE * sizeof(uint8_t));


	uint8_t key[AES_KEYLEN];
	uint8_t iv[AES_BLOCKLEN];
	genArray(AES_KEYLEN, key, &RANDSEED2);
	genArray(AES_BLOCKLEN, iv, &RANDSEED3);
	long bufSize, returnSize;

	genArray(ASIZE, a, &RANDSEED);


	if (!QUIET) {
		printf("\n");
		printf("****************\n");
		printf("** ECB TESTS **\n");
		printf("****************\n");


		printDesc("cpu AES ECB");
		bufSize = AES::CPU::encryptECB(key, a, b, ASIZE);
		printElapsedTime(AES::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "Encrypt (std::chrono Measured)");
		returnSize = AES::CPU::decryptECB(key, b, c, bufSize);
		printElapsedTime(AES::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "Decrypt (std::chrono Measured)");
		printCmpResult(ASIZE, a, c);

		printDesc("gpu AES ECB");
	}
	bufSize = AES::GPU::encryptECB(key, a, d, ASIZE);
	if (QUIET) {
		printf("%f\n", AES::GPU::timer().getGpuElapsedTimeForPreviousOperation());
	}
	else {
		printElapsedTime(AES::GPU::timer().getGpuElapsedTimeForPreviousOperation(), "Encrypt (std::chrono Measured)");
		printCmpResult(ASIZE, b, d);//might get wonky with the extra padding bytes, but should get most of the way there
	}
	returnSize = AES::GPU::decryptECB(key, d, e, bufSize);
	if (QUIET) {
		printf("%f\n", AES::GPU::timer().getGpuElapsedTimeForPreviousOperation());
	}
	else {
		printElapsedTime(AES::GPU::timer().getGpuElapsedTimeForPreviousOperation(), "Decrypt (std::chrono Measured)");
		printCmpResult(ASIZE, a, e);
	}
	zeroArray(bufSize, d);
	zeroArray(bufSize, e);


	if (!QUIET) {
		printf("\n");
		printf("****************\n");
		printf("** CTR TESTS **\n");
		printf("****************\n");

		printDesc("cpu AES CTR");
		bufSize = AES::CPU::encryptCTR(key, iv, a, b, ASIZE);
		printElapsedTime(AES::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "Encrypt (std::chrono Measured)");
		returnSize = AES::CPU::decryptCTR(key, iv, b, c, bufSize);
		printElapsedTime(AES::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "Decrypt (std::chrono Measured)");
		printCmpResult(ASIZE, a, c);

		printDesc("gpu AES CTR");
	}
	bufSize = AES::GPU::encryptCTR(key, iv, a, d, ASIZE);
	if (QUIET) {
		printf("%f\n", AES::GPU::timer().getGpuElapsedTimeForPreviousOperation());
	}
	else {
		printElapsedTime(AES::GPU::timer().getGpuElapsedTimeForPreviousOperation(), "Encrypt (std::chrono Measured)");
		printCmpResult(ASIZE, b, d);//might get wonky with the extra padding bytes, but should get most of the way there
	}
	returnSize = AES::GPU::decryptCTR(key, iv, d, e, bufSize);
	if (QUIET) {
		printf("%f\n", AES::GPU::timer().getGpuElapsedTimeForPreviousOperation());
	}
	else {
		printElapsedTime(AES::GPU::timer().getGpuElapsedTimeForPreviousOperation(), "Decrypt (std::chrono Measured)");
		printCmpResult(ASIZE, a, e);



		system("pause"); // stop Win32 console from closing on exit
	}
	free(a);
	free(b);
	free(c);
	free(d);
	free(e);
}
