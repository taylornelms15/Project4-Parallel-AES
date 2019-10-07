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
#include <string>
#include <fstream>
#include "testing_helpers.hpp"

#define OFFPOT 1

const unsigned int RANDSEED = 0xbad1bad2;
const unsigned int RANDSEED2 = 0x0123fed5;
const unsigned int RANDSEED3 = 0x56781234;

const int SIZE = 1 << 20; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
#if OFFPOT
static int ASIZE = NPOT;
#else
static int ASIZE = SIZE;
#endif

static std::string infile = std::string();
static std::string keycontents = std::string();

/**
Reads a file in.
Important Note: mallocs the memory for the file
*/
uint64_t readFile(std::string filepath, uint8_t** dest) {
	std::ifstream ifs(filepath, std::ifstream::binary);
	std::string contents = std::string((std::istreambuf_iterator<char>(ifs)),
										(std::istreambuf_iterator<char>()));

	uint64_t contentSize = contents.size();
	*dest = (uint8_t*)malloc(contentSize * sizeof(uint8_t));
	memcpy(*dest, contents.data(), contentSize * sizeof(uint8_t));

	ifs.close();
	return contentSize;
}

void writeFile(std::string infilePath, const uint8_t* buffer, uint64_t size, std::string suffix) {
	std::ofstream outfile(std::string(infilePath).append(suffix), std::ofstream::binary);
	const char* mybuf = (const char*)buffer;
	outfile.write(mybuf, size);
	outfile.close();
}//writeFile

uint8_t hexToVal(char hex) {
	if (hex >= '0' && hex <= '9') return (uint8_t)(hex - '0');
	if (hex >= 'a' && hex <= 'f') return (uint8_t)(hex - 'a' + 10);
	if (hex >= 'A' && hex <= 'F') return (uint8_t)(hex - 'A' + 10);
	return 0;
}

void keyStringToByteBuffer(std::string keystring, uint8_t* buffer) {
	for (int i = 0; i < AES_SIZE; i++) {
		char chi = keystring.at(2 * i + 0);
		char clo = keystring.at(2 * i + 1);
		buffer[i] = (hexToVal(chi) << 4) | hexToVal(clo);
	}
}


int main(int argc, char* argv[]) {
	//parse arguments
	cxxopts::Options options("aes_test", "Program to test parallel AES encryption/decryption");
	options.add_options()
		("k,keysize", "Size of AES key", cxxopts::value<int>()->default_value("256"))
		("b,blocksize", "Size of cuda blocks to use", cxxopts::value<int>()->default_value("256"))
		("n,blocksperthread", "How many AES blocks each thread processes", cxxopts::value<int>()->default_value("1"))
		("y,sharedmemKey", "Whether storing key in shared memory")
		("s,sharedmemSBox", "Whether storing sbox in shared memory")
		("c,constantMem", "Whether storing relevant components in constant memory")
		("p,parameter", "Whether we are pulling keys and sboxes from a passed parameter")
		("q,quiet", "Runs just the GPU tests, outputs just the timings for automatic consumption")
		("i,infile", "Specifies a file to encrypt", cxxopts::value<std::string>())
		("x,key", "Specified a hex key to use; must be correct length", cxxopts::value<std::string>())
		;
	auto result = options.parse(argc, argv);
	bool sharedmemKey = false; bool sharedmemSBox = false; 
	bool parameter = false; bool quiet = false; bool constant = false;
	int blocksize; int keysize; int blocksperthread;
	sharedmemKey	= result.count("sharedmemKey") > 0;
	sharedmemSBox	= result.count("sharedmemSBox") > 0;
	parameter		= result.count("parameter") > 0;
	quiet			= result.count("quiet") > 0;
	constant		= result.count("constantMem") > 0;
	blocksize		= result["blocksize"].as<int>();
	keysize			= result["keysize"].as<int>();
	blocksperthread = result["blocksperthread"].as<int>();
	if (result.count("infile")) {
		infile = result["infile"].as<std::string>();
	}
	if (result.count("key")) {
		keycontents = result["key"].as<std::string>();
	}

	ingestCommandLineOptions(keysize, blocksize, blocksperthread, sharedmemKey, sharedmemSBox, parameter, quiet, constant);

	if (keycontents.size() && (keycontents.size() != AES_SIZE / 4)) {//num bytes, times 2
		printf("Error with given key: not correct size. Using random instead.");
		keycontents = std::string();
	}

	uint8_t* a;
	if (infile.size()) {
		ASIZE = readFile(infile, &a);
	}
	else {
		a = (uint8_t*)malloc(ASIZE * sizeof(uint8_t));
		genArray(ASIZE, a, &RANDSEED);
	}

	//Run tests

	uint8_t* b = (uint8_t*)malloc((ASIZE + AES_BLOCKLEN) * sizeof(uint8_t));
	uint8_t* c = (uint8_t*)malloc(ASIZE * sizeof(uint8_t));
	uint8_t* d = (uint8_t*)malloc((ASIZE + AES_BLOCKLEN) * sizeof(uint8_t));
	uint8_t* e = (uint8_t*)malloc(ASIZE * sizeof(uint8_t));


	uint8_t key[AES_KEYLEN];
	uint8_t iv[AES_BLOCKLEN];
	genArray(AES_BLOCKLEN, iv, &RANDSEED3);
	long bufSize, returnSize;

	if (keycontents.size()) {
		keyStringToByteBuffer(keycontents, a);
	}
	else {
		genArray(AES_KEYLEN, key, &RANDSEED2);
	}

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

	if (infile.size()) {
		writeFile(infile, d, bufSize, ".ecb");
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
	if (infile.size()) {
		writeFile(infile, d, bufSize, ".ctr");
	}

	free(a);
	free(b);
	free(c);
	free(d);
	free(e);
}
