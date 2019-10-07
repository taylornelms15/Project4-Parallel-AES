#!/usr/bin/env python3
import sys
import os
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt

MEMORY_MODES = ["Global", "Shared_Memory", "Parameter", "Shared_Key_Only", "Shared_SBox_Only", "Global_Constant"]

baseExecutablePath = "../build/Release/cis565_aes_test.exe"
imgPath = "../img/blade_model.jpg"
imgRoot = "../img"

def transformPicture(picturePath):
    #read it in python-style, output it to binary, encrypt, read those in python-style, output as pictures
    folder, basename = os.path.split(picturePath)
    basename, extension = os.path.splitext(basename)

    img = cv2.imread(picturePath,cv2.IMREAD_COLOR)
    shape = img.shape
    size = shape[0] * shape[1] * shape[2]

    binaryToSave = list(np.reshape(img, (size)))

    binaryDest = os.path.join(folder, basename + ".bin")
    with open(binaryDest, "wb") as f:
        for byteval in binaryToSave:
            f.write(byteval)


    args = [baseExecutablePath, "-q", "-i", binaryDest]

    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    #print("Output after image transform: " + str(resultString))

    ecbDest = binaryDest + ".ecb"
    ctrDest = binaryDest + ".ctr"

    with open(ecbDest, "rb") as f:
        ecbContents = f.read()
    with open(ctrDest, "rb") as f:
        ctrContents = f.read()

    ecbContents = ecbContents[:size]
    ecbImage = np.frombuffer(ecbContents, dtype="uint8")
    ecbImage = np.reshape(ecbImage, shape)
    ctrContents = ctrContents[:size]
    ctrImage = np.frombuffer(ctrContents, dtype="uint8")
    ctrImage = np.reshape(ctrImage, shape)

    ecbImgDest = os.path.join(folder, basename + "_ecb" + extension)
    ctrImgDest = os.path.join(folder, basename + "_ctr" + extension)

    cv2.imwrite(ecbImgDest, ecbImage)
    cv2.imwrite(ctrImgDest, ctrImage)

    os.remove(binaryDest)
    os.remove(ecbDest)
    os.remove(ctrDest)

def getAverageRunResults(memoryMode, blockSize = 256, keySize = 256, blocksPerThread = 1):

    runResults = list(getRunResults(memoryMode, blockSize, keySize, blocksPerThread))

    for i in range(1,20):
        nextResults = list(getRunResults(memoryMode, blockSize, keySize, blocksPerThread))
        for j in range(4):
            runResults[j] *= i / (i + 1.0)
            runResults[j] += nextResults[j] * (1.0 / (i + 1))

    print(runResults)
    return runResults
            

def getRunResults(memoryMode, blockSize = 256, keySize = 256, blocksPerThread = 1):
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

    args.append("-n")
    args.append(str(blocksPerThread))

    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout.decode("ascii")
    ecbEnc, ecbDec, ctrEnc, ctrDec = resultString.split()
    ecbEnc = float(ecbEnc)
    ecbDec = float(ecbDec)
    ctrEnc = float(ctrEnc)
    ctrDec = float(ctrDec)

    return (ecbEnc, ecbDec, ctrEnc, ctrDec)

def testMemoryModes(blockSize = 256, keySize = 256, blocksPerThread = 1):
    results = {}
    for memoryMode in MEMORY_MODES:
        resultSet = getAverageRunResults(memoryMode, blockSize)
        results[memoryMode] = resultSet

    return results


def main():
   
    #transformPicture(imgPath)

    results = testMemoryModes(256, 192)
    for k, v in results.items():
        print("%s\t%s" % (k, v))



if __name__ == "__main__":
    main()
