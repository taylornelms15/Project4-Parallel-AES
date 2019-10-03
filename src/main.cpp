/**
 * @file      main.cpp
 * @brief     AES Test Program
 * @authors   Taylor Nelms
 * @date      2019
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <aes/cpu.h>
#include <aes/naive.h>
#include "testing_helpers.hpp"

#define OFFPOT 1

const unsigned int RANDSEED = 0xbad1bad2;

const int SIZE = 1 << 20; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
#if OFFPOT
const int ASIZE = NPOT;
#else
const int ASIZE = SIZE
#endif


int main(int argc, char* argv[]) {
	uint8_t* a = (uint8_t*)malloc(ASIZE * sizeof(uint8_t));
	uint8_t* b = (uint8_t*)malloc((ASIZE + AES_BLOCKLEN) * sizeof(uint8_t));
	uint8_t* c = (uint8_t*)malloc(ASIZE * sizeof(uint8_t));
	uint8_t* d = (uint8_t*)malloc((ASIZE + AES_BLOCKLEN) * sizeof(uint8_t));
	uint8_t* e = (uint8_t*)malloc(ASIZE * sizeof(uint8_t));


	uint8_t key[AES_KEYLEN];
	uint8_t iv[AES_BLOCKLEN];
	genArray(AES_KEYLEN, key, &RANDSEED);
	genArray(AES_BLOCKLEN, iv, &RANDSEED);
	long bufSize, returnSize;

	genArray(ASIZE, a, &RANDSEED);


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
	bufSize = AES::Naive::encryptECB(key, a, d, ASIZE);
	printElapsedTime(AES::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "Encrypt (std::chrono Measured)");
	printCmpResult(ASIZE, b, d);//might get wonky with the extra padding bytes, but should get most of the way there
	returnSize = AES::Naive::decryptECB(key, d, e, bufSize);
	printElapsedTime(AES::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "Decrypt (std::chrono Measured)");
	printCmpResult(ASIZE, a, e);



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
	bufSize = AES::Naive::encryptCTR(key, iv, a, d, ASIZE);
	printElapsedTime(AES::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "Encrypt (std::chrono Measured)");
	printCmpResult(ASIZE, b, d);//might get wonky with the extra padding bytes, but should get most of the way there
	returnSize = AES::Naive::decryptCTR(key, iv, d, e, bufSize);
	printElapsedTime(AES::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "Decrypt (std::chrono Measured)");
	printCmpResult(ASIZE, a, e);


    system("pause"); // stop Win32 console from closing on exit
	free(a);
	free(b);
	free(c);
	free(d);
	free(e);
}
