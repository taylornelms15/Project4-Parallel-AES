CUDA AES Encryption
===================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Taylor Nelms
  * [LinkedIn](https://www.linkedin.com/in/taylor-k-7b2110191/), [twitter](https://twitter.com/nelms_taylor), etc.
* Tested on: Windows 10, Intel i3 Coffee Lake 4-core 3.6GHz processor, 16GB RAM, NVidia GeForce GTX1650 4GB

## Motivation

Encryption is a pervasive element of cybersecurity; however, despite needing to encrypt vast amounts of data, there is a relative lack of parallelizable options to do so. Some of that is algorithmic in nature; various encryption algorithms are data-dependent in an unbroken chain from the first bit to the last. However, there are algorithms that allow for a data-parallel approach to encryption; in particular, the ECB and CTR encryption modes of the AES algorithm allow for such an operation.

This project will endeavor to optimize a parallel approach to AES encryption on the GPU via the CUDA programming frameworks. I will attempt to tweak any and all configuration parameters and memory access patterns to allow for highly efficient encryption/decryption leveraging the GPU's capabilities.

## Notes for Implementation

sbox, rsbox: 256-size byte arrays for lookups. can compare global memory to shared memory for them. Rcon-11 bytes, same thing. (this is for key expansion)
In reality - key expansion should probably be on CPU; only happens once. (this is the initializing-context bit)

The ECB mode does encryption on the plaintext directly, 16-byte block by 16-byte block
The CTR mode does encryption on the counter IV (and it is being modified as we go), and XOR-ing that with the plaintext. Effectively, it's using the IV and the key to create a chain of noise, and just XOR'ing that. (thus, why you never want to repeat an IV and key combo; if one thing is broken, or its plaintext contents are known, somebody has the whole set of info)

For the encryption/decryption steps:
* The Roundkey (240 bytes) is the value shared to all potential block processes.
    * Can compare having it as a parameter (direct to registers...?), global memory, constant memory, and pulling it into shared memory
* Similarly, the substitution box (SBox) operates with the same behavior. This is a 256-byte substitution table. (The inverse, rsbox, behaves the same)
* In terms of parallelization, the state is operated on sequentially, both within a round of processing and between rounds.
    * It is not feasible to parallelize between these steps
    * While it may be feasible to parallelize bits within the steps (such as subBytes), I'm going to propose that the step would not be worth the overhead (and programmer headache), as a pair of threads could just as easily be working on more blocks as doing just one. Additionally, this cuts down the need to share the state between threads, which could get significant.


## References

* CPU baseline implementation taken from kokke's [tiny-AES-c](https://github.com/kokke/tiny-AES-c.git) implementation
* Command-line parsing using [cxxopts](https://github.com/jarro2783/cxxopts)
