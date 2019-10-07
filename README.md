CUDA AES Encryption
===================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Taylor Nelms
  * [LinkedIn](https://www.linkedin.com/in/taylor-k-7b2110191/), [twitter](https://twitter.com/nelms_taylor)
* Tested on: Windows 10, Intel i3 Coffee Lake 4-core 3.6GHz processor, 16GB RAM, NVidia GeForce GTX1650 4GB

![Encryption/Decryption Times on CPU and GPU by Input Size](img/InputSizeChart.png)

## Motivation

Encryption is a pervasive element of cybersecurity; however, despite needing to encrypt vast amounts of data, there is a relative lack of parallelizable options to do so. Some of that is algorithmic in nature; various encryption algorithms are data-dependent in an unbroken chain from the first bit to the last. However, there are algorithms that allow for a data-parallel approach to encryption; in particular, the ECB and CTR encryption modes of the AES algorithm allow for such an operation.

This project will endeavor to optimize a parallel approach to AES encryption on the GPU via the CUDA programming frameworks. I will attempt to tweak any and all configuration parameters and memory access patterns to allow for highly efficient encryption/decryption leveraging the GPU's capabilities.

## The AES Algorithm

The **Advanced Encryption Standard**, or **AES**, is an encryption algorithm adopted by NIST in 2001. It was developed, and subsequently adopted, to fill holes and vulnerabilities in the Data Encryption Standard (which dated back to 1977).

It is a **block cipher** that operates with a symmetric key. This means that it operates on fixed-length blocks of data; in this case, **16-byte** blocks. The idea behind symmetric-key encryption is that a single key is used to both encrypt and decrypt the data; this is in contrast to an algorithm like RSA, which uses two related, asymmetric keys (containing both a public and private component) to encrypt data between two parties.

The AES standard is defined for key lengths of **128**, **192**, and **256-bit** key sizes (16, 24, and 32 bytes). A well-used AES with a 128-bit key is considered effectively unbreakable via brute force methods; the larger key sizes are even more so.

### AES Modes

There are a few different ways to implement a block cipher like AES. The algorithm itself only applies to a single block of plaintext and how it combines with a key to produce the ciphertext; for larger data, we can approach the encryption in different fashions.

#### ECB

The easiest mode is **Electronic Codebook**, or **ECB**. This simply encrypts each block of plaintext using the same key in the same manner, and appends the results together.

This method is naively parallelizable, as the same operation is applied to each block of plaintext; it is one of the modes implemented in this project.

This has the disadvantage that, for repeated plaintext data, the ciphertext will be similarly repeated. As such, even though retrieving the exact plaintext may be impossible, the structure of the plaintext data may still be visible.

TODO: create image of the penguin, or similar

#### CTR

**Counter** mode, or **CTR**, takes a different approach. It requires an **Initialization Vector**, or **IV**, along with the key and plantext to operate. This IV is the same size as the block length (16 bytes). This IV is encrypted with the key, and then **XOR**'d with the first block of plaintext to produce the first block of ciphertext. The IV is then incremented for the next block, and the process repeats.

Notably, this means that for any IV, we can determine what the IV will be at a given block in the ciphertext. As such, we can easily parallelize this approach, with each kernel operating on what it knows the IV will be at its location.

It is important when using this mode to not re-use the same IV with any key. Every combination of key and IV will create a unique string of data, which, when **XOR**'d with a string of plaintext, produces the ciphertext. This can lead to information leaks, where, if you know both the ciphertext and plaintext for a given key-iv pair, you can then produce the encryption string for that pair with a simple xor operation.

One slight advantage of this approach, though, is that encryption and decryption use the same operation, as each simply construct the encryption-string and xor the plaintext/ciphertext with it.

#### CBC

**Cipher Block Chaining**, or **CBC** mode, is considered slightly more secure than CTR. Instead of simply incrementing a previous block's **IV** to get the IV for the next block, it operates by **XOR**ing the plaintext with the first IV, encrypting that, and then using the resulting ciphertext as the IV for the next step.

This has the advantage that each block of ciphertext is dependent on the plaintext for that block, the key, the initial IV, and every block of plaintext that came before it.

Unfortunately for us, this means it is impossible to data-parallelize, and it is not considered in this GPU implementation.

### Padding

For data that does not fit within the block size, data ends up padded to meet the block size requirements. For my implementation, I used [PKCS#7](https://en.wikipedia.org/wiki/Padding_(cryptography)#PKCS#5_and_PKCS#7) padding (though other methods could be used); in brief, this involves figuring out how many bytes short of a block the data is, and adding bytes containing that value to the end until it is full. (If we are already at a block boundary, an extra block is added.)

For example, if our data contained a final partial block with only four bytes in it, we would pad the remaining 12 (`0x0C`) to the end in the following manner: "0xXX 0xYY 0xZZ 0xWW **0x0C 0x0C 0x0C 0x0C 0x0C 0x0C 0x0C 0x0C 0x0C 0x0C 0x0C 0x0C**".

### The Algorithm

#### Rounds

The AES algorithm does a set of operations to the plaintext across a number of rounds. Each size key uses a different number of rounds: `10` rounds for `128`-bit keys, `12` rounds for `192`-bit keys, and `14` rounds for `256`-bit keys. This number of rounds roughly correlates to the amount of computational work done overall; below, we see the encryption/decryption time for different key lengths on a CPU.

![Effect of key size on encryption/decryption time on a CPU](img/KeylenChartCPU.png)

The algorithm is applied in series to a working state of the plaintext block; each step (summarized below) operates on the block in-place. This working state is traditionally treated as a column-major `4x4` byte matrix.

#### Steps

Each round does four main steps (with some differences across the final round): **Substitute Bytes**, **Shift Rows**, **Mix Columns**, and **Add Key**.

##### Substitute Bytes

Each byte is substituted for a different value. This is most easily done via a 256-entry lookup table. (The inverse of this step is done using the inverse of this table.) These are referred to for my implementation as the **S-box** and **RS-box**, respectively. Given how many lookups are done in this table, the issue of where we store it is an important performance consideration.

##### Shift Rows

While remembering that the state is in a column-major matrix, the issue of shifting rows becomes slightly less memory-trivial. Nevertheless, this step is simply a matter of rotating each matrix row to the left some number of times; `0` times for the first row, `1` for the second, `2` for the third, and `3` for the final row. The inverse operation is the same, but in the opposite direction.

##### Mix Columns

One of the more mathematically complex steps, this produces each byte of column output as a combination of each other byte in the column. It uses something tantalizingly close to matrix multiplication, but is unfortunately closer to matrix multiplication using finite-field arithmetic, which limited my ability to make use of CUDA primitives to expedite the process.

##### Add Key

The key itself is, at the beginning of the algorithm, transformed into a key schedule (the **RoundKey** in my implementation, which is relevant in the same way the **S-box** is). Depending on which round we are in, we simply **XOR** the state with the key schedule at a particular point.

Since the expanded key is between `176` and `240` bytes long (depending on our key size), and is used multiple times and by all blocks in the same way, the method by which we store and reference it is an important consideration to this project.

##### Further Reading

For a more in-depth discussion about the structure of the algorithm, I highly recommend [this pdf from Purdue](https://engineering.purdue.edu/kak/compsec/NewLectures/Lecture8.pdf), which proved useful when implementing this project.

## My Implementation

### Basic Approach

Though it may have been technically feasible to parallelize or pipeline dataflow within the encryption of each block, I instead elected to perform the encryption of each block using a single kernel thread; that way, I could eliminate data dependencies between threads, and (ideally) each warp could simultaneously operate on 32 blocks without interruption.

I allow for one thread to encrypt/decrypt multiple AES blocks as well, to see if there is any advantage to reducing the thread count overall and instead have each thread do more work.

#### Memory

The input and output buffers are, of course, very large, and each thread will need to access a different part of them. As such, global memory seems the reasonable place to keep them.

However, there are a few pieces of data that are constant to each thread. One are the tables for byte substitution, **sbox** and **rsbox**. Another is the key schedule itself, **roundkey**; it is precomputed on the CPU, but then used by each kernel without being modified at all.

As such, we have a few options of how to put these chunks of data (containing up to `752` total bytes) somewhere that each thread can access them.

* Access each table from global memory
* Put each table into global constant memory
* Pull the contents of the tables into shared memory
    * Implementation allows for putting the key, the substitution boxes, or both into shared memory
* Wrap the tables into a struct and pass them directly to each kernel via a parameter to the `__global__ void` function

#### CUDA Block Size

As we know, each CUDA block can contain a different number of threads. I allow for dynamically configurable block sizes, and can thereby analyze the performance impacts thereof.

## Performance Analysis

![Encryption/Decryption Times on CPU and GPU by Input Size](img/InputSizeChart.png)

We see from this chart that, for data ranges between roughly `250KB` and `250MB`, we acheive a nearly 150-times speedup for our GPU implementation over a CPU implementation, just using global memory for our constant tables and expanded key.

However, we have identified a number of tunable parameters that we can use to adjust performance time for GPU operations; as such, further analysis will consider GPU implementations against each other, rather than compared to the CPU implementations.

##### Why the longer time for ECB decryption?

It is worth noting, before we go further, that while ECB encryption, CTR encryption, and CTR decryption all have functionally identical execution times, the same cannot be said for ECB decryption, which often runs close to twice as long. Why is that? Most of the encryption/decryption steps are nearly identical and very symmetric; however, the **Mix Columns** step contains a huge computational load for decryption that did not exist for encryption.

TODO: explain the difference, and show the Speed of Light graph that points to it as well.

## References

* CPU baseline implementation and key expansion implementation taken from kokke's [tiny-AES-c](https://github.com/kokke/tiny-AES-c.git) implementation
* Command-line parsing using [cxxopts](https://github.com/jarro2783/cxxopts)
