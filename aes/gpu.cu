#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "common.h"
#include "gpu.h"
#include <math.h>

namespace AES {
    namespace GPU {
        using AES::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		//#####################
		// PRECOMPUTED TABLES
		//#####################

		static const uint8_t sbox[256] = {
			0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
			0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
			0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
			0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
			0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
			0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
			0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
			0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
			0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
			0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
			0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
			0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
			0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
			0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
			0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
			0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 };

		static const uint8_t rsbox[256] = {
		  0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
		  0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
		  0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
		  0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
		  0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
		  0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
		  0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
		  0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
		  0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
		  0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
		  0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
		  0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
		  0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
		  0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
		  0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
		  0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d };

		static const uint8_t Rcon[11] = {
			 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };

		//#####################
		// FUNCTION DEFINITIONS
		//#####################
		
		__device__ void copyDataToSharedMemory(uint8_t* shared, uint8_t** roundkey, uint8_t** sbox, uint8_t** rsbox, int roundKeySize, uint8_t sharedMemFlags);


		//####################
		// LITTLE HELPERS
		//####################

		__host__ __device__ uint64_t getPaddedLength(uint64_t currentLength) {
			uint64_t newLen = ((currentLength / AES_BLOCKLEN) + 1) * AES_BLOCKLEN;
			return newLen;
		}

#define GF_MAGIC 0x1B

		__host__ __device__  uint8_t gfmult2(uint8_t x) {
			uint8_t result = (x >> 7) ? (x << 1) ^ GF_MAGIC : (x << 1);
			return result;
		}

		__host__ __device__  uint8_t gfmult3(uint8_t x) {
			uint8_t result = ((x >> 7) ? (x << 1) ^ GF_MAGIC : (x << 1)) ^ x;
			//uint8_t result = gfmult2(x) ^ x;
			return result;
		}

		/**
		The larger numbers might be inefficient; perhaps go about the operations differently?
		May not be an efficient way to do it, unfortunately
		*/
		__host__ __device__  uint8_t gfmult9(uint8_t x) {
			uint8_t result = gfmult2(gfmult2(gfmult2(x))) ^ x;
			return result;
		}
		__host__ __device__  uint8_t gfmult11(uint8_t x) {
			uint8_t result = gfmult2(gfmult2(gfmult2(x)) ^ x) ^ x;
			return result;
		}
		__host__ __device__  uint8_t gfmult13(uint8_t x) {
			uint8_t result = gfmult2(gfmult2(gfmult2(x) ^ x)) ^ x;
			return result;
		}
		__host__ __device__  uint8_t gfmult14(uint8_t x) {
			uint8_t result = gfmult2(gfmult2(gfmult2(x) ^ x) ^ x);
			return result;
		}

		/**
		GF-field multiplication
		This is not super-secure, but should be reasonably sensible to debug
		*/
		__host__ __device__  uint8_t gfmult(uint8_t x, uint8_t y) {
			switch (y) {
			case 0x02:
				return gfmult2(x);
				break;
			case 0x03:
				return gfmult3(x);
				break;
			case 0x09:
				return gfmult9(x);
				break;
			case 0x0B:
				return gfmult11(x);
				break;
			case 0x0D:
				return gfmult13(x);
				break;
			case 0x0E:
				return gfmult14(x);
				break;
			default:
				return 0x00;
				break;
			}
		}

		//####################
		// GLOBAL MEMORY
		//####################

		uint8_t* d_input;
		uint8_t* d_output;
		uint8_t* d_roundkey;
		uint8_t* d_iv;
		uint8_t* d_sbox;
		uint8_t* d_rsbox;

		/**
		Need a container for the constant memory because alignment can be a problem when casting the roundkey to uint32_t for XOR'ing
		*/
		typedef struct __align__(256) AlignedBox {
			uint8_t data[256];
		} AlignedBox;

		__constant__ AlignedBox csbox;
		__constant__ AlignedBox crsbox;
		__constant__ AlignedBox croundkey;

		#define Nb 4//always 4 columns for stae
		__constant__ uint8_t Nkvars[2];//first one is Nk, second one is Nr (Nk not used on device, I later learned)


		//####################
		// KERNEL FUNCTIONS
		//####################
				
		/**
		Encryption substitute bytes step (can double as the decryption step)
		Modifies state in place
		Table wants to be a pointer to a 256-byte array containing the substitution table
		*/
		__device__ void subBytes(State* state, const uint8_t* table) {
			for (uint8_t i = 0; i < 4; i++) {
#if USING_VECTORS
				state->data[i].x = table[state->data[i].x];
				state->data[i].y = table[state->data[i].y];
				state->data[i].z = table[state->data[i].z];
				state->data[i].w = table[state->data[i].w];
#else
				state->data[i][0] = table[state->data[i][0]];
				state->data[i][1] = table[state->data[i][1]];
				state->data[i][2] = table[state->data[i][2]];
				state->data[i][3] = table[state->data[i][3]];
#endif
			}//for each col
		}//subBytes

		/**
		Encryption shift-rows step
		Modifies state in-place
		*/
		__device__ void shiftRows(State* state) {
			uint8_t temp;
#if USING_VECTORS
			//row index 1
			temp = state->data[0].y;
			state->data[0].y = state->data[1].y;
			state->data[1].y = state->data[2].y;
			state->data[2].y = state->data[3].y;
			state->data[3].y = temp;
			//row index 2
				//swap one
			temp = state->data[0].z;
			state->data[0].z = state->data[2].z;
			state->data[2].z = temp;
				//swap the other
			temp = state->data[1].z;
			state->data[1].z = state->data[3].z;
			state->data[3].z = temp;
			//row index 3
			temp = state->data[3].w;
			state->data[3].w = state->data[2].w;
			state->data[2].w = state->data[1].w;
			state->data[1].w = state->data[0].w;
			state->data[0].w = temp;
#else
			//row index 1
			temp = state->data[0][1];
			state->data[0][1] = state->data[1][1];
			state->data[1][1] = state->data[2][1];
			state->data[2][1] = state->data[3][1];
			state->data[3][1] = temp;
			//row index 2
				//swap one
			temp = state->data[0][2];
			state->data[0][2] = state->data[2][2];
			state->data[2][2] = temp;
			//swap the other
			temp = state->data[1][2];
			state->data[1][2] = state->data[3][2];
			state->data[3][2] = temp;
			//row index 3
			temp = state->data[3][3];
			state->data[3][3] = state->data[2][3];
			state->data[2][3] = state->data[1][3];
			state->data[1][3] = state->data[0][3];
			state->data[0][3] = temp;
#endif
		}//shiftRows

		/**
		Decryption shift-rows step
		Modifies state in-place
		*/
		__device__ void invShiftRows(State* state) {
			uint8_t temp;
#if USING_VECTORS
			//row index 1
			temp = state->data[0].y;
			state->data[0].y = state->data[3].y;
			state->data[3].y = state->data[2].y;
			state->data[2].y = state->data[1].y;
			state->data[1].y = temp;
			//row index 2
				//swap one
			temp = state->data[0].z;
			state->data[0].z = state->data[2].z;
			state->data[2].z = temp;
				//swap the other
			temp = state->data[1].z;
			state->data[1].z = state->data[3].z;
			state->data[3].z = temp;
			//row index 3
			temp = state->data[3].w;
			state->data[3].w = state->data[0].w;
			state->data[0].w = state->data[1].w;
			state->data[1].w = state->data[2].w;
			state->data[2].w = temp;
#else
			//row index 1
			temp = state->data[0][1];
			state->data[0][1] = state->data[3][1];
			state->data[3][1] = state->data[2][1];
			state->data[2][1] = state->data[1][1];
			state->data[1][1] = temp;
			//row index 2
				//swap one
			temp = state->data[0][2];
			state->data[0][2] = state->data[2][2];
			state->data[2][2] = temp;
				//swap the other
			temp = state->data[1][2];
			state->data[1][2] = state->data[3][2];
			state->data[3][2] = temp;
			//row index 3
			temp = state->data[3][3];
			state->data[3][3] = state->data[0][3];
			state->data[0][3] = state->data[1][3];
			state->data[1][3] = state->data[2][3];
			state->data[2][3] = temp;
#endif
		}//invShiftRows

		/**
		Encryption mix-columns step
		Modifies state in-place
		*/
		__device__ void mixColumns(State* state) {
			for (uint8_t i = 0; i < 4; i++) {
#if USING_VECTORS
				uchar4 oldvals = state->data[i];
				state->data[i].x = gfmult2(oldvals.x) ^ gfmult3(oldvals.y) ^ oldvals.z ^ oldvals.w;
				state->data[i].y = oldvals.x ^ gfmult2(oldvals.y) ^ gfmult3(oldvals.z) ^ oldvals.w;
				state->data[i].z = oldvals.x ^ oldvals.y ^ gfmult2(oldvals.z) ^ gfmult3(oldvals.w);
				state->data[i].w = gfmult3(oldvals.x) ^ oldvals.y ^ oldvals.z ^ gfmult2(oldvals.w);
				/*
				state->data[i].x = gfmult(oldvals.x, 0x02) ^ gfmult(oldvals.y, 0x03) ^ oldvals.z ^ oldvals.w;
				state->data[i].y = oldvals.x ^ gfmult(oldvals.y, 0x02) ^ gfmult(oldvals.z, 0x03) ^ oldvals.w;
				state->data[i].z = oldvals.x ^ oldvals.y ^ gfmult(oldvals.z, 0x02) ^ gfmult(oldvals.w, 0x03);
				state->data[i].w = gfmult(oldvals.x, 0x03) ^ oldvals.y ^ oldvals.z ^ gfmult(oldvals.w, 0x02);
				*/
#else
				uint8_t oldvals[4] = { state->data[i][0], state->data[i][1], state->data[i][2], state->data[i][3] };
				state->data[i][0] = gfmult2(oldvals[0]) ^ gfmult3(oldvals[1]) ^ oldvals[2] ^ oldvals[3];
				state->data[i][1] = oldvals[0] ^ gfmult2(oldvals[1]) ^ gfmult3(oldvals[2]) ^ oldvals[3];
				state->data[i][2] = oldvals[0] ^ oldvals[1] ^ gfmult2(oldvals[2]) ^ gfmult3(oldvals[3]);
				state->data[i][3] = gfmult3(oldvals[0]) ^ oldvals[1] ^ oldvals[2] ^ gfmult2(oldvals[3]);
#endif
			}//for
		}//mixColumns

		/**
		Decryption mix-columns step
		Modifies state in-place
		*/
		__device__ void invMixColumns(State* state) {
			for (uint8_t i = 0; i < 4; i++) {
#if USING_VECTORS
				uchar4 oldvals = state->data[i];
				state->data[i].x = gfmult14(oldvals.x) ^ gfmult11(oldvals.y) ^ gfmult13(oldvals.z) ^ gfmult9(oldvals.w);
				state->data[i].y = gfmult9(oldvals.x) ^ gfmult14(oldvals.y) ^ gfmult11(oldvals.z) ^ gfmult13(oldvals.w);
				state->data[i].z = gfmult13(oldvals.x) ^ gfmult9(oldvals.y) ^ gfmult14(oldvals.z) ^ gfmult11(oldvals.w);
				state->data[i].w = gfmult11(oldvals.x) ^ gfmult13(oldvals.y) ^ gfmult9(oldvals.z) ^ gfmult14(oldvals.w);
#else
				uint8_t oldvals[4] = { state->data[i][0], state->data[i][1], state->data[i][2], state->data[i][3] };
				state->data[i][0] = gfmult14(oldvals[0]) ^ gfmult11(oldvals[1]) ^ gfmult13(oldvals[2]) ^ gfmult9(oldvals[3]);
				state->data[i][1] = gfmult9(oldvals[0]) ^ gfmult14(oldvals[1]) ^ gfmult11(oldvals[2]) ^ gfmult13(oldvals[3]);
				state->data[i][2] = gfmult13(oldvals[0]) ^ gfmult9(oldvals[1]) ^ gfmult14(oldvals[2]) ^ gfmult11(oldvals[3]);
				state->data[i][3] = gfmult11(oldvals[0]) ^ gfmult13(oldvals[1]) ^ gfmult9(oldvals[2]) ^ gfmult14(oldvals[3]);
#endif
			}//for
		}//mixColumns

		/**
		Encryption add-key step
		Modifies state in-place
		*/
		__device__ void addKey(State* state, const uint8_t* roundKey, uint8_t roundNum) {

			for (uint8_t i = 0; i < 4; i++) {
				unsigned rkbase = (roundNum * Nb * 4) + (i * Nb);
#if USING_VECTORS

				*reinterpret_cast<uint32_t*>(&state->data[i]) = 
					*reinterpret_cast<uint32_t*>(&state->data[i]) ^
					*reinterpret_cast<const uint32_t*>(&roundKey[rkbase]);
				//state->data[i].x ^= roundKey[rkbase + 0];
				//state->data[i].y ^= roundKey[rkbase + 1];
				//state->data[i].z ^= roundKey[rkbase + 2];
				//state->data[i].w ^= roundKey[rkbase + 3];
				
#else
				*reinterpret_cast<uint32_t*>(&state->data[i]) = 
					*reinterpret_cast<uint32_t*>(&state->data[i]) ^ 
					*reinterpret_cast<const uint32_t*>(&roundKey[rkbase]);
#endif
			}

		}//addKey

		__device__ void incrementIvByValue(uint8_t* iv, uint64_t addition) {
			uint64_t rawLow =   ((uint64_t)iv[15]) << 0 |
								((uint64_t)iv[14]) << 8 |
								((uint64_t)iv[13]) << 16 |
								((uint64_t)iv[12]) << 24 |
								((uint64_t)iv[11]) << 32 |
								((uint64_t)iv[10]) << 40 |
								((uint64_t)iv[9]) << 48 |
								((uint64_t)iv[8]) << 56;
			uint64_t rawHi =    ((uint64_t)iv[7]) << 0 |
								((uint64_t)iv[6]) << 8 |
								((uint64_t)iv[5]) << 16 |
								((uint64_t)iv[4]) << 24 |
								((uint64_t)iv[3]) << 32 |
								((uint64_t)iv[2]) << 40 |
								((uint64_t)iv[1]) << 48 |
								((uint64_t)iv[0]) << 56;
			uint64_t newLow = rawLow + addition;
			uint64_t newHi	= rawHi + (newLow < rawLow ? 1 : 0);
			for (int i = 0; i < 8; i++) {
				iv[15 - i] = (uint8_t)((newLow >> 8 * i) & 0xff);
			}
			for (int i = 0; i < 8; i++) {
				iv[7 - i] = (uint8_t)((newHi >> 8 * i) & 0xff);
			}
			
		}//incrementIvByValue

		//######################
		// ENCRYPTION GLOBALS
		//######################

		/**
		Encrypts a single 16-byte block
		*/
		__device__ void Encrypt(uint8_t* idata, uint8_t* odata,
							uint8_t* roundkey, uint8_t* sbox, uint8_t* rsbox) {
			//set up current state
			State curState;
			memcpy(&curState.data[0], idata + 0, 4 * sizeof(uint8_t));
			memcpy(&curState.data[1], idata + 4, 4 * sizeof(uint8_t));
			memcpy(&curState.data[2], idata + 8, 4 * sizeof(uint8_t));
			memcpy(&curState.data[3], idata + 12, 4 * sizeof(uint8_t));

			//encrypt
			addKey(&curState, roundkey, 0);
			for (uint8_t i = 1; i < Nkvars[1]; i++) {
				subBytes(&curState, sbox);
				shiftRows(&curState);
				mixColumns(&curState);
				addKey(&curState, roundkey, i);
			}
			subBytes(&curState, sbox);
			shiftRows(&curState);
			addKey(&curState, roundkey, Nkvars[1]);

			//copy to output
			memcpy(&odata[0], &curState.data[0], 4 * sizeof(uint8_t));
			memcpy(&odata[4], &curState.data[1], 4 * sizeof(uint8_t));
			memcpy(&odata[8], &curState.data[2], 4 * sizeof(uint8_t));
			memcpy(&odata[12], &curState.data[3], 4 * sizeof(uint8_t));
		}

		__device__ void Decrypt(uint8_t* idata, uint8_t* odata,
					uint8_t* roundkey, uint8_t* sbox, uint8_t* rsbox) {
			//set up current state
			State curState;
			memcpy(&curState.data[0], idata + 0, 4 * sizeof(uint8_t));
			memcpy(&curState.data[1], idata + 4, 4 * sizeof(uint8_t));
			memcpy(&curState.data[2], idata + 8, 4 * sizeof(uint8_t));
			memcpy(&curState.data[3], idata + 12, 4 * sizeof(uint8_t));

			//decrypt
			addKey(&curState, roundkey, Nkvars[1]);
			for (uint8_t i = Nkvars[1] - 1; i > 0; i--) {
				invShiftRows(&curState);
				subBytes(&curState, rsbox);
				addKey(&curState, roundkey, i);
				invMixColumns(&curState);
			}
			invShiftRows(&curState);
			subBytes(&curState, rsbox);
			addKey(&curState, roundkey, 0);

			//copy to output
			memcpy(&odata[0], &curState.data[0], 4 * sizeof(uint8_t));
			memcpy(&odata[4], &curState.data[1], 4 * sizeof(uint8_t));
			memcpy(&odata[8], &curState.data[2], 4 * sizeof(uint8_t));
			memcpy(&odata[12], &curState.data[3], 4 * sizeof(uint8_t));
		}


		__global__ void encryptUsingGlobalMem(uint8_t* idata, uint8_t* odata, 
											uint8_t* roundkey, uint8_t* sbox, uint8_t* rsbox, 
											uint64_t bytesToEncrypt, int numAblocksPerThread, bool constantMem) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int byteOffset = index * BYTES_PER_ABLOCK * numAblocksPerThread;
			if (byteOffset >= bytesToEncrypt) return;
			if (constantMem) {
				sbox		= csbox.data;
				rsbox		= crsbox.data;
				roundkey	= croundkey.data;
			}

			for (int i = 0; i < numAblocksPerThread; i++) {
				int thisOffset = byteOffset + BYTES_PER_ABLOCK * i;
				uint8_t* myInputStartPoint = idata + thisOffset;
				uint8_t* myOutputStartPoint = odata + thisOffset;

				Encrypt(myInputStartPoint, myOutputStartPoint, roundkey, sbox, rsbox);
			}//for each block we're encrypting: do it!
		}

		__global__ void encryptCTRUsingGlobalMem(uint8_t* idata, uint8_t* odata,
											uint8_t* roundkey, uint8_t* iv,
											uint8_t* sbox, uint8_t* rsbox,
											uint64_t bytesToEncrypt, int numAblocksPerThread, bool constantMem) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			long byteOffset = index * BYTES_PER_ABLOCK * numAblocksPerThread;
			if (byteOffset >= bytesToEncrypt) return;
			if (constantMem) {
				sbox		= csbox.data;
				rsbox		= crsbox.data;
				roundkey	= croundkey.data;
			}

			uint8_t myIv[AES_BLOCKLEN];

			for (int i = 0; i < numAblocksPerThread; i++) {
				int blockOffset = index * numAblocksPerThread + i;
				memcpy(myIv, iv, AES_BLOCKLEN * sizeof(uint8_t));
				incrementIvByValue(myIv, blockOffset);

				int thisOffset = byteOffset + BYTES_PER_ABLOCK * i;
				uint8_t* myInputStartPoint = idata + thisOffset;
				uint8_t* myOutputStartPoint = odata + thisOffset;

				Encrypt(myIv, myIv, roundkey, sbox, rsbox);
				for (uint8_t j = 0; j < AES_BLOCKLEN; j++) {
					myOutputStartPoint[j] = myIv[j] ^ myInputStartPoint[j];
				}
			}//for each block we're encrypting: do it!
		}

		__global__ void decryptUsingGlobalMem(uint8_t* idata, uint8_t* odata,
											uint8_t* roundkey, uint8_t* sbox, uint8_t* rsbox,
											uint64_t bytesToEncrypt, int numAblocksPerThread, bool constantMem) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int byteOffset = index * BYTES_PER_ABLOCK * numAblocksPerThread;
			if (byteOffset >= bytesToEncrypt) return;
			if (constantMem) {
				sbox		= csbox.data;
				rsbox		= crsbox.data;
				roundkey	= croundkey.data;
			}

			for (int i = 0; i < numAblocksPerThread; i++) {
				int thisOffset = byteOffset + BYTES_PER_ABLOCK * i;
				uint8_t* myInputStartPoint = idata + thisOffset;
				uint8_t* myOutputStartPoint = odata + thisOffset;

				Decrypt(myInputStartPoint, myOutputStartPoint, roundkey, sbox, rsbox);
			}//for each block we're encrypting: do it!
		}

		__global__ void encryptUsingSharedMem(uint8_t* idata, uint8_t* odata,
									uint8_t* roundkey, uint8_t* sbox, uint8_t* rsbox,
									uint64_t bytesToEncrypt, int numAblocksPerThread, 
									int roundKeySize, uint8_t sharedMemFlags) {
			extern __shared__ uint8_t shared[];
			//copy into shared memory
			copyDataToSharedMemory(shared, &roundkey, &sbox, &rsbox, roundKeySize, sharedMemFlags);

			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int byteOffset = index * BYTES_PER_ABLOCK * numAblocksPerThread;
			if (byteOffset >= bytesToEncrypt) return;


			for (int i = 0; i < numAblocksPerThread; i++) {
				int thisOffset = byteOffset + BYTES_PER_ABLOCK * i;
				uint8_t* myInputStartPoint = idata + thisOffset;
				uint8_t* myOutputStartPoint = odata + thisOffset;

				Encrypt(myInputStartPoint, myOutputStartPoint, roundkey, sbox, rsbox);
			}//for each block we're encrypting: do it!
		}

		__global__ void encryptCTRUsingSharedMem(uint8_t* idata, uint8_t* odata,
											uint8_t* roundkey, uint8_t* iv,
											uint8_t* sbox, uint8_t* rsbox,
											uint64_t bytesToEncrypt, int numAblocksPerThread,
											int roundKeySize, uint8_t sharedMemFlags) {
			extern __shared__ uint8_t shared[];
			//copy into shared memory
			copyDataToSharedMemory(shared, &roundkey, &sbox, &rsbox, roundKeySize, sharedMemFlags);

			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int byteOffset = index * BYTES_PER_ABLOCK * numAblocksPerThread;
			if (byteOffset >= bytesToEncrypt) return;

			uint8_t myIv[AES_BLOCKLEN];

			for (int i = 0; i < numAblocksPerThread; i++) {
				int blockOffset = index * numAblocksPerThread + i;
				memcpy(myIv, iv, AES_BLOCKLEN * sizeof(uint8_t));
				incrementIvByValue(myIv, blockOffset);

				int thisOffset = byteOffset + BYTES_PER_ABLOCK * i;
				uint8_t* myInputStartPoint = idata + thisOffset;
				uint8_t* myOutputStartPoint = odata + thisOffset;

				Encrypt(myIv, myIv, roundkey, sbox, rsbox);
				for (uint8_t j = 0; j < AES_BLOCKLEN; j++) {
					myOutputStartPoint[j] = myIv[j] ^ myInputStartPoint[j];
				}
			}//for each block we're encrypting: do it!
		}

		__global__ void decryptUsingSharedMem(uint8_t* idata, uint8_t* odata,
									uint8_t* roundkey, uint8_t* sbox, uint8_t* rsbox,
									uint64_t bytesToEncrypt, int numAblocksPerThread,
			int roundKeySize, uint8_t sharedMemFlags) {
			extern __shared__ uint8_t shared[];
			//copy into shared memory
			copyDataToSharedMemory(shared, &roundkey, &sbox, &rsbox, roundKeySize, sharedMemFlags);

			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int byteOffset = index * BYTES_PER_ABLOCK * numAblocksPerThread;
			if (byteOffset >= bytesToEncrypt) return;


			for (int i = 0; i < numAblocksPerThread; i++) {
				int thisOffset = byteOffset + BYTES_PER_ABLOCK * i;
				uint8_t* myInputStartPoint = idata + thisOffset;
				uint8_t* myOutputStartPoint = odata + thisOffset;

				Decrypt(myInputStartPoint, myOutputStartPoint, roundkey, sbox, rsbox);
			}//for each block we're encrypting: do it!
		}

		__global__ void encryptUsingParameter(uint8_t* idata, uint8_t* odata,
									KeyBox keyBox,
									uint64_t bytesToEncrypt, int numAblocksPerThread) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int byteOffset = index * BYTES_PER_ABLOCK * numAblocksPerThread;
			if (byteOffset >= bytesToEncrypt) return;

			for (int i = 0; i < numAblocksPerThread; i++) {
				int thisOffset = byteOffset + BYTES_PER_ABLOCK * i;
				uint8_t* myInputStartPoint = idata + thisOffset;
				uint8_t* myOutputStartPoint = odata + thisOffset;

				Encrypt(myInputStartPoint, myOutputStartPoint, keyBox.roundkey, keyBox.sbox, keyBox.rsbox);
			}//for each block we're encrypting: do it!
		}

		__global__ void encryptCTRUsingParameter(uint8_t* idata, uint8_t* odata,
											KeyBox keyBox, uint8_t* iv,
											uint64_t bytesToEncrypt, int numAblocksPerThread) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int byteOffset = index * BYTES_PER_ABLOCK * numAblocksPerThread;
			if (byteOffset >= bytesToEncrypt) return;

			uint8_t myIv[AES_BLOCKLEN];

			for (int i = 0; i < numAblocksPerThread; i++) {
				int blockOffset = index * numAblocksPerThread + i;
				memcpy(myIv, iv, AES_BLOCKLEN * sizeof(uint8_t));
				incrementIvByValue(myIv, blockOffset);

				int thisOffset = byteOffset + BYTES_PER_ABLOCK * i;
				uint8_t* myInputStartPoint = idata + thisOffset;
				uint8_t* myOutputStartPoint = odata + thisOffset;

				Encrypt(myIv, myIv, keyBox.roundkey, keyBox.sbox, keyBox.rsbox);
				for (uint8_t j = 0; j < AES_BLOCKLEN; j++) {
					myOutputStartPoint[j] = myIv[j] ^ myInputStartPoint[j];
				}
			}//for each block we're encrypting: do it!
		}

		__global__ void decryptUsingParameter(uint8_t* idata, uint8_t* odata,
									KeyBox keyBox,
									uint64_t bytesToEncrypt, int numAblocksPerThread) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int byteOffset = index * BYTES_PER_ABLOCK * numAblocksPerThread;
			if (byteOffset >= bytesToEncrypt) return;

			for (int i = 0; i < numAblocksPerThread; i++) {
				int thisOffset = byteOffset + BYTES_PER_ABLOCK * i;
				uint8_t* myInputStartPoint = idata + thisOffset;
				uint8_t* myOutputStartPoint = odata + thisOffset;

				Decrypt(myInputStartPoint, myOutputStartPoint, keyBox.roundkey, keyBox.sbox, keyBox.rsbox);
			}//for each block we're encrypting: do it!
		}

		//######################
		// MEMORY HELPERS
		//######################

		__device__ void copyDataToSharedMemory(uint8_t* shared, uint8_t** roundkey, uint8_t** sbox, uint8_t** rsbox, int roundKeySize, uint8_t sharedMemFlags) {
			int sboxoffset = 0;
			if (sharedMemFlags & SHAREDMEM_KEYMASK) {
				int numElementsICopy = (roundKeySize + blockDim.x - 1) / blockDim.x;
				for (int i = 0; i < numElementsICopy; i++) {
					int indexICopy = blockDim.x * i + threadIdx.x;
					if (indexICopy >= roundKeySize) continue;
					shared[indexICopy] = (*roundkey)[indexICopy];
				}
				__syncthreads();
				*roundkey = &shared[0];
				sboxoffset += roundKeySize;
			}//if key in shared memory
			if (sharedMemFlags & SHAREDMEM_SBOXMASK) {
				int numElementsICopy = (256 + blockDim.x - 1) / blockDim.x;
				for (int i = 0; i < numElementsICopy; i++) {
					int indexICopy = blockDim.x * i + threadIdx.x;
					if (indexICopy >= 256) continue;
					shared[sboxoffset + indexICopy] = (*sbox)[indexICopy];
					shared[sboxoffset + indexICopy + 256] = (*rsbox)[indexICopy];
				}
				__syncthreads();
				*sbox = &shared[sboxoffset];
				*rsbox = &shared[sboxoffset + 256];
			}
		}

		void copyNkToConstantMemory() {
			uint8_t Nk, Nr;
			switch (AES_SIZE) {
			case 256:
				Nk = 8;
				Nr = 14;
				break;
			case 192:
				Nk = 6;
				Nr = 12;
				break;
			case 128:
				Nk = 4;
				Nr = 10;
				break;
			default:
				break;
			}
			uint8_t transfer[2] = { Nk, Nr };
			cudaMemcpyToSymbol(Nkvars, &transfer, 2 * sizeof(uint8_t));
			checkCUDAError("CudaMemcpySymbol");

		}//copyNkToConstantMemory

		int getSharedMemSize(uint8_t* sharedMemMask) {
			*sharedMemMask = 0;
			int total = 0;
			if (SHAREDMEM_KEY) {
				*sharedMemMask |= SHAREDMEM_KEYMASK;
				total += AES_KEY_EXP_SIZE;
			}
			if (SHAREDMEM_SBOX) {
				*sharedMemMask |= SHAREDMEM_SBOXMASK;
				total += 512;
			}
			return total;
		}

		void initGlobalMemIO(uint64_t size) {
			cudaMalloc((void**)&d_input, size * sizeof(uint8_t));
			cudaMalloc((void**)&d_output, size * sizeof(uint8_t));
			checkCUDAError("CudaMalloc");
		}


		void initGlobalMemKey(const uint8_t* key) {
			uint8_t Nk, Nr;
			switch (AES_SIZE) {
			case 256:
				Nk = 8;
				Nr = 14;
				break;
			case 192:
				Nk = 6;
				Nr = 12;
				break;
			case 128:
				Nk = 4;
				Nr = 10;
				break;
			default:
				break;
			}
			uint8_t* roundkey = (uint8_t*)malloc(AES_KEY_EXP_SIZE * sizeof(uint8_t));
			expandKey(key, roundkey, Nk, Nr);



			if (CONSTANTMEM) {
				cudaMemcpyToSymbol(croundkey.data, roundkey, AES_KEY_EXP_SIZE * sizeof(uint8_t));
				checkCUDAError("cudamemcpysymbol");
			}
			else {
				cudaMalloc(&d_roundkey, AES_KEY_EXP_SIZE * sizeof(uint8_t));
				checkCUDAError("cudaMalloc");
				cudaMemcpy(d_roundkey, roundkey, AES_KEY_EXP_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);
				checkCUDAError("cudamemcpy");
			}

			free(roundkey);
		}

		void initGlobalMemIv(const uint8_t* iv) {
			cudaMalloc(&d_iv, AES_BLOCKLEN * sizeof(uint8_t));
			checkCUDAError("cudaMalloc");
			cudaMemcpy(d_iv, iv, AES_BLOCKLEN * sizeof(uint8_t), cudaMemcpyHostToDevice);
			checkCUDAError("cudamemcpy");
		}

		void initKeyBox(const uint8_t* key, KeyBox* keyBox) {
			uint8_t Nk, Nr;
			switch (AES_SIZE) {
			case 256:
				Nk = 8;
				Nr = 14;
				break;
			case 192:
				Nk = 6;
				Nr = 12;
				break;
			case 128:
				Nk = 4;
				Nr = 10;
				break;
			default:
				break;
			}
			uint8_t* roundkey = (uint8_t*)malloc(AES_KEY_EXP_SIZE * sizeof(uint8_t));
			expandKey(key, roundkey, Nk, Nr);

			memcpy(keyBox->roundkey, roundkey, AES_KEY_EXP_SIZE * sizeof(uint8_t));
			memcpy(keyBox->sbox, sbox, 256);
			memcpy(keyBox->rsbox, rsbox, 256);

			free(roundkey);
		}

		void initGlobalMemSbox() {

			if (CONSTANTMEM) {
				cudaMemcpyToSymbol(csbox.data, sbox, 256 * sizeof(uint8_t));
				cudaMemcpyToSymbol(crsbox.data, rsbox, 256 * sizeof(uint8_t));
				checkCUDAError("cudamemcpysymbol");
			}
			else {
				cudaMalloc(&d_sbox, 256 * sizeof(uint8_t));
				cudaMalloc(&d_rsbox, 256 * sizeof(uint8_t));
				checkCUDAError("cudamalloc");
				cudaMemcpy(d_sbox, sbox, 256 * sizeof(uint8_t), cudaMemcpyHostToDevice);
				cudaMemcpy(d_rsbox, rsbox, 256 * sizeof(uint8_t), cudaMemcpyHostToDevice);
				checkCUDAError("cudamemcpy");
			}
		}

		void deinitGlobalMemIO() {
			cudaFree(d_input);
			cudaFree(d_output);
			checkCUDAError("Cudafree");
		}

		void deinitGlobalMemKey() {
			if (!CONSTANTMEM) {
				cudaFree(d_roundkey);
				checkCUDAError("cudafree");
			}
		}

		void deinitGlobalMemIv() {
			cudaFree(d_iv);
			checkCUDAError("cudafree");
		}

		void deinitGlobalMemSbox() {
			if (!CONSTANTMEM) {
				cudaFree(d_sbox);
				cudaFree(d_rsbox);
				checkCUDAError("cudafree");
			}
		}

		//######################
		// ENCRYPTION MAIN FUNCS
		//######################

		dim3 getNumBlocks(uint64_t dataSize, int ablocksPerThread) {
			int tpb = BLOCKSIZE;
			long numBlocks = dataSize / AES_BLOCKLEN;

			int numAblocksHandledByBlock = ablocksPerThread * tpb;
			long bpg = (numBlocks + numAblocksHandledByBlock - 1) / numAblocksHandledByBlock;
			return dim3(bpg);
		}

		long encryptECB(const uint8_t* key, const uint8_t* input, uint8_t* output, uint64_t bufferLength) {
			uint64_t paddedLength = getPaddedLength(bufferLength);
			uint8_t lenDiffArray[AES_BLOCKLEN] = {};
			uint8_t lenDiff = (uint8_t)(paddedLength - bufferLength);
			for (uint8_t i = 0; i < AES_BLOCKLEN; i++) {
				lenDiffArray[i] = lenDiff;
			}

			//malloc space for padded input/output
			initGlobalMemIO(paddedLength);
			//copy input
			cudaMemcpy(d_input, input, bufferLength * sizeof(uint8_t), cudaMemcpyHostToDevice);
			checkCUDAError("CudaMemcpy");
			//pad input
			cudaMemcpy(d_input + bufferLength, lenDiffArray, lenDiff, cudaMemcpyHostToDevice);
			checkCUDAError("CudaMemcpy");
			copyNkToConstantMemory();
			initGlobalMemKey(key);
			initGlobalMemSbox();

			//kernel initialization params
			int numAblocksPerThread = ABLOCKS_PER_THREAD;
			dim3 tpb = dim3(BLOCKSIZE);
			dim3 bpg = getNumBlocks(paddedLength, numAblocksPerThread);

			timer().startGpuTimer();

			if (USING_SHAREDMEM) {
				uint8_t sharedMemMask = 0x00;
				int sharedmemSize = getSharedMemSize(&sharedMemMask);
				encryptUsingSharedMem <<<bpg, tpb, sharedmemSize >>> (d_input, d_output,
					d_roundkey, d_sbox, d_rsbox,
					paddedLength, numAblocksPerThread,
					AES_KEY_EXP_SIZE, sharedMemMask);
			}
			else if (USING_PARAMETER) {
				KeyBox keyBox;
				initKeyBox(key, &keyBox);
				encryptUsingParameter <<<bpg, tpb >>> (d_input, d_output,
							keyBox,
							bufferLength, numAblocksPerThread);
			}
			else {
				encryptUsingGlobalMem <<<bpg, tpb >>> (d_input, d_output,
					d_roundkey, d_sbox, d_rsbox,
					paddedLength, numAblocksPerThread, CONSTANTMEM);
			}

			timer().endGpuTimer();

			//copy output
			cudaMemcpy(output, d_output, paddedLength * sizeof(uint8_t), cudaMemcpyDeviceToHost);
			checkCUDAError("CudaMemcpy");

			//free input/output
			deinitGlobalMemKey();
			deinitGlobalMemSbox();
			deinitGlobalMemIO();
			return (long)paddedLength;
		}

		long decryptECB(const uint8_t* key, const uint8_t* input, uint8_t* output, uint64_t bufferLength) {
			//malloc space for padded input/output
			initGlobalMemIO(bufferLength);
			//copy input
			cudaMemcpy(d_input, input, bufferLength * sizeof(uint8_t), cudaMemcpyHostToDevice);
			checkCUDAError("CudaMemcpy");
			copyNkToConstantMemory();
			initGlobalMemKey(key);
			initGlobalMemSbox();

			//kernel initialization params
			int numAblocksPerThread = ABLOCKS_PER_THREAD;
			dim3 tpb = dim3(BLOCKSIZE);
			dim3 bpg = getNumBlocks(bufferLength, numAblocksPerThread);

			timer().startGpuTimer();

			if (USING_SHAREDMEM) {
				uint8_t sharedMemMask = 0x00;
				int sharedmemSize = getSharedMemSize(&sharedMemMask);
				decryptUsingSharedMem<<<bpg, tpb, sharedmemSize >>>(d_input, d_output,
								d_roundkey, d_sbox, d_rsbox,
								bufferLength, numAblocksPerThread,
								AES_KEY_EXP_SIZE, sharedMemMask);
			}
			else if (USING_PARAMETER) {
				KeyBox keyBox;
				initKeyBox(key, &keyBox);
				decryptUsingParameter<<<bpg, tpb >>>(d_input, d_output,
					keyBox,
					bufferLength, numAblocksPerThread);
			}
			else {
				decryptUsingGlobalMem<<<bpg, tpb >>>(d_input, d_output,
								d_roundkey, d_sbox, d_rsbox,
								bufferLength, numAblocksPerThread, CONSTANTMEM);
			}

			timer().endGpuTimer();

			//copy output
			cudaMemcpy(output, d_output, bufferLength * sizeof(uint8_t), cudaMemcpyDeviceToHost);
			checkCUDAError("CudaMemcpy");

			//unpad output
			uint64_t unpadLength = AES::Common::unpadData(output, bufferLength);

			deinitGlobalMemKey();
			deinitGlobalMemSbox();
			deinitGlobalMemIO();
			return (long)unpadLength;
		}

		long encryptCTR(const uint8_t* key, const uint8_t* iv, const uint8_t* input, uint8_t* output, uint64_t bufferLength) {
			uint64_t paddedLength = getPaddedLength(bufferLength);
			uint8_t lenDiffArray[AES_BLOCKLEN] = {};
			uint8_t lenDiff = (uint8_t)(paddedLength - bufferLength);
			for (uint8_t i = 0; i < AES_BLOCKLEN; i++) {
				lenDiffArray[i] = lenDiff;
			}

			//malloc space for padded input/output
			initGlobalMemIO(paddedLength);
			//copy input
			cudaMemcpy(d_input, input, bufferLength * sizeof(uint8_t), cudaMemcpyHostToDevice);
			checkCUDAError("CudaMemcpy");
			//pad input
			cudaMemcpy(d_input + bufferLength, lenDiffArray, lenDiff, cudaMemcpyHostToDevice);
			checkCUDAError("CudaMemcpy");
			copyNkToConstantMemory();
			initGlobalMemKey(key);
			initGlobalMemSbox();
			initGlobalMemIv(iv);

			//kernel initialization params
			int numAblocksPerThread = ABLOCKS_PER_THREAD;
			dim3 tpb = dim3(BLOCKSIZE);
			dim3 bpg = getNumBlocks(paddedLength, numAblocksPerThread);

			timer().startGpuTimer();

			if (USING_SHAREDMEM) {
				uint8_t sharedMemMask = 0x00;
				int sharedmemSize = getSharedMemSize(&sharedMemMask);
				encryptCTRUsingSharedMem << <bpg, tpb, sharedmemSize >> > (d_input, d_output,
					d_roundkey, d_iv,
					d_sbox, d_rsbox,
					paddedLength, numAblocksPerThread,
					AES_KEY_EXP_SIZE, sharedMemMask);
			}
			else if (USING_PARAMETER) {
				KeyBox keyBox;
				initKeyBox(key, &keyBox);
				encryptCTRUsingParameter << <bpg, tpb >> > (d_input, d_output,
					keyBox, d_iv,
					bufferLength, numAblocksPerThread);
			}
			else {
				encryptCTRUsingGlobalMem <<<bpg, tpb >>> (d_input, d_output,
					d_roundkey, d_iv, d_sbox, d_rsbox,
					paddedLength, numAblocksPerThread, CONSTANTMEM);
			}

			timer().endGpuTimer();

			//copy output
			cudaMemcpy(output, d_output, paddedLength * sizeof(uint8_t), cudaMemcpyDeviceToHost);
			checkCUDAError("CudaMemcpy");

			//free input/output
			deinitGlobalMemKey();
			deinitGlobalMemSbox();
			deinitGlobalMemIO();
			deinitGlobalMemIv();
			return (long)paddedLength;
		}

		long decryptCTR(const uint8_t* key, const uint8_t* iv, const uint8_t* input, uint8_t* output, uint64_t bufferLength) {
			//malloc space for padded input/output
			initGlobalMemIO(bufferLength);
			//copy input
			cudaMemcpy(d_input, input, bufferLength * sizeof(uint8_t), cudaMemcpyHostToDevice);
			checkCUDAError("CudaMemcpy");
			copyNkToConstantMemory();
			initGlobalMemKey(key);
			initGlobalMemSbox();
			initGlobalMemIv(iv);

			//kernel initialization params
			int numAblocksPerThread = ABLOCKS_PER_THREAD;
			dim3 tpb = dim3(BLOCKSIZE);
			dim3 bpg = getNumBlocks(bufferLength, numAblocksPerThread);

			timer().startGpuTimer();

			if (USING_SHAREDMEM) {
				uint8_t sharedMemMask = 0x00;
				int sharedmemSize = getSharedMemSize(&sharedMemMask);
				encryptCTRUsingSharedMem << <bpg, tpb, sharedmemSize >> > (d_input, d_output,
					d_roundkey, d_iv,
					d_sbox, d_rsbox,
					bufferLength, numAblocksPerThread,
					AES_KEY_EXP_SIZE, sharedMemMask);
			}
			else if (USING_PARAMETER) {
				KeyBox keyBox;
				initKeyBox(key, &keyBox);
				encryptCTRUsingParameter << <bpg, tpb >> > (d_input, d_output,
					keyBox, d_iv,
					bufferLength, numAblocksPerThread);
			}
			else {
				encryptCTRUsingGlobalMem << <bpg, tpb >> > (d_input, d_output,
					d_roundkey, d_iv, d_sbox, d_rsbox,
					bufferLength, numAblocksPerThread, CONSTANTMEM);
			}

			timer().endGpuTimer();

			//copy output
			cudaMemcpy(output, d_output, bufferLength * sizeof(uint8_t), cudaMemcpyDeviceToHost);
			checkCUDAError("CudaMemcpy");

			//unpad output
			uint64_t unpadLength = AES::Common::unpadData(output, bufferLength);

			deinitGlobalMemKey();
			deinitGlobalMemIv();
			deinitGlobalMemSbox();
			deinitGlobalMemIO();
			return (long)unpadLength;
		}


		void expandKey(const uint8_t* key, uint8_t* expanded, uint8_t Nk, uint8_t Nr) {
			uint8_t tempa[4]; // Used for the column/row operations
			uint8_t j, k;

			for (uint8_t i = 0; i < Nk; i++) {
				expanded[(i * 4) + 0] = key[(i * 4) + 0];
				expanded[(i * 4) + 1] = key[(i * 4) + 1];
				expanded[(i * 4) + 2] = key[(i * 4) + 2];
				expanded[(i * 4) + 3] = key[(i * 4) + 3];
			}//first four words are the same

			  // All other round keys are found from the previous round keys.
			for (uint8_t i = Nk; i < Nb * (Nr + 1); ++i)
			{
				{
					k = (i - 1) * 4;
					tempa[0] = expanded[k + 0];
					tempa[1] = expanded[k + 1];
					tempa[2] = expanded[k + 2];
					tempa[3] = expanded[k + 3];

				}

				if (i % Nk == 0)
				{
					// This function shifts the 4 bytes in a word to the left once.
					// [a0,a1,a2,a3] becomes [a1,a2,a3,a0]

					// Function RotWord()
					{
						const uint8_t u8tmp = tempa[0];
						tempa[0] = tempa[1];
						tempa[1] = tempa[2];
						tempa[2] = tempa[3];
						tempa[3] = u8tmp;
					}

					// SubWord() is a function that takes a four-byte input word and 
					// applies the S-box to each of the four bytes to produce an output word.

					// Function Subword()
					{
						tempa[0] = sbox[tempa[0]];
						tempa[1] = sbox[tempa[1]];
						tempa[2] = sbox[tempa[2]];
						tempa[3] = sbox[tempa[3]];
					}

					tempa[0] = tempa[0] ^ Rcon[i / Nk];
				}
				if (AES_SIZE == 256) {
					if (i % Nk == 4)
					{
						// Function Subword()
						{
							tempa[0] = sbox[tempa[0]];
							tempa[1] = sbox[tempa[1]];
							tempa[2] = sbox[tempa[2]];
							tempa[3] = sbox[tempa[3]];
						}
					}
				}
				j = i * 4; k = (i - Nk) * 4;
				expanded[j + 0] = expanded[k + 0] ^ tempa[0];
				expanded[j + 1] = expanded[k + 1] ^ tempa[1];
				expanded[j + 2] = expanded[k + 2] ^ tempa[2];
				expanded[j + 3] = expanded[k + 3] ^ tempa[3];
			}

		}
    }
}
