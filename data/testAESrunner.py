#!/usr/bin/env python3
import sys
import os
import subprocess

import pdb

MEMORY_MODES = ["normal", "shared", "parameter", "sharedKey", "sharedSBox", "constant"]

baseExecutablePath = "../build/Release/cis565_aes_test.exe"

def getRunResults(memoryMode, blockSize = 256, keySize = 256):
    args = [baseExecutablePath, "-q"]

    if memoryMode == MEMORY_MODES[0]:
        pass
    elif memoryMode == MEMORY_MODES[1]:
        args.append("-ys")
    elif memoryMode == MEMORY_MODES[2]:
        args.append("-p")
    elif memoryMode == MEMORY_MODES[3]:
        args.append("-y")
    elif memoryMode == MEMORY_MODES[4]:
        args.append("-s")
    elif memoryMode == MEMORY_MODES[5]:
        args.append("-c")

    args.append("-b")
    args.append(str(blockSize))

    args.append("-k")
    args.append(str(keySize))

    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    ecbEnc, ecbDec, ctrEnc, ctrDec = resultString.split()
    ecbEnc = float(ecbEnc)
    ecbDec = float(ecbDec)
    ctrEnc = float(ctrEnc)
    ctrDec = float(ctrDec)

    return (ecbEnc, ecbDec, ctrEnc, ctrDec)

def testMemoryModes(blockSize = 256, keySize = 256):
    results = {}
    for memoryMode in MEMORY_MODES:
        resultSet = getRunResults(memoryMode, blockSize)
        results[memoryMode] = resultSet

    return results


def main():
   
    results = getRunResults("sharedKey", 512)

    results = testMemoryModes(256, 192)
    for k, v in results.items():
        print("%s\t%s" % (k, v))



if __name__ == "__main__":
    main()
