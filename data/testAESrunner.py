#!/usr/bin/env python3
import sys
import os
import re
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt

MEMORY_MODES = ["Global", "Shared_Memory", "Parameter", "Shared_Key_Only", "Shared_SBox_Only", "Global_Constant"]
BLOCK_SIZES = [32, 64, 128, 192, 256, 384, 512, 768, 1024]
NUM_BLOCKS_PER_THREAD = [1, 2, 4, 8]

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

def getGroupedBarChart(runResultsTuples, tupleLabels, groupTitles, xTitle, yTitle, chartTitle, chartSubtitle = None, filename = None):
    """
    @param runResultsTuples: list of tuples (one tuple per bar)
    @param tupleLabels: How the above tuples should be labeled
    @param groupTitles: how each list element in runResultsTuples should be titled (turns to column groups)
    @param xTitle: Title for X axis
    @param yTitle: Title for Y axis
    @param chartTitle: Title for chart
    @param chartSubtitle: Subtitle for chart
    @param filename: output file path, if saving png directly
    """
    numColumns = len(runResultsTuples)
    numPerCol = len(runResultsTuples[0])

    fig, ax = plt.subplots()
    ind = np.arange(numColumns)
    width = 1.0 / (numPerCol + 1)

    plist = []

    for i in range(numPerCol):
        setVals = [x[i] for x in runResultsTuples]
        plist.append(ax.bar(ind + i * width, setVals, width, align = "edge"))

    plt.suptitle(chartTitle, fontweight = "bold")
    if chartSubtitle:
        ax.set_title(chartSubtitle)
    ax.set_xlabel(xTitle, fontweight = "bold")
    ax.set_ylabel(yTitle, fontweight = "bold")
    ax.set_xticks(ind + width * numPerCol / 2.0)
    ax.set_xticklabels(groupTitles)

    ax.legend(tuple([x[0] for x in plist]), tupleLabels)

    if not filename:
        plt.show()
    else:
        pass
        #TODO: remember how to save

    

def getAverageRunResults(memoryMode, blockSize = 256, keySize = 256, blocksPerThread = 1):

    runResults = list(getRunResults(memoryMode, blockSize, keySize, blocksPerThread))

    for i in range(1,10):
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

def testBlockSize(keySize = 256, memMode = MEMORY_MODES[0]):
    resultTuples = []
    for blockSize in BLOCK_SIZES:
        bptResults = []
        for blocksPerThread in NUM_BLOCKS_PER_THREAD:
            ctrTime = getAverageRunResults(memMode, blockSize, keySize, blocksPerThread)[2]
            bptResults.append(ctrTime)
        resultTuples.append(tuple(bptResults))


    tupleTitles = [str(x) + " aes blocks per thread" for x in NUM_BLOCKS_PER_THREAD]
    groupTitles = [str(x) for x in BLOCK_SIZES]
    xTitle = "Number of CUDA threads per block"
    yTitle = "Completion Time (ms)"
    chartTitle = "Performance of different workload distributions"
    chartSubtitle = "16.78MB Input, %s-bit AES, CTR Encryption time" % (keySize)


    getGroupedBarChart(resultTuples, tupleTitles, groupTitles,\
            xTitle, yTitle, chartTitle, chartSubtitle)

    


def testMemoryModes(blockSize = 256, keySize = 256, blocksPerThread = 1):
    results = {}
    resultTuples = []
    for memoryMode in MEMORY_MODES:
        if memoryMode == MEMORY_MODES[2]:
            continue#skip parameter
        resultSet = getAverageRunResults(memoryMode, blockSize)
        resultTuples.append(resultSet)
        results[memoryMode] = resultSet

    groupTitles = [re.sub("_", " ", x) for x in MEMORY_MODES if x is not MEMORY_MODES[2]]
#    groupTitles = [re.sub("_", " ", x) for x in MEMORY_MODES]
    tupleTitles = ["ECB Encrypt", "ECB Decrypt", "CTR Encrypt", "CTR Decrypt"]
    xTitle = "Memory Access Mode"
    yTitle = "Completion Time (ms)"
    chartTitle = "Performance of Different Memory Access Modes"
    chartSubtitle = "16.78MB Input, %s-bit AES, %s threads per block" % (keySize, blockSize)

    getGroupedBarChart(resultTuples, tupleTitles, groupTitles,\
            xTitle, yTitle, chartTitle, chartSubtitle)

    return results


def main():
   
    #transformPicture(imgPath)

    #results = testMemoryModes(64, 192)
    results = testBlockSize(192)
    for k, v in results.items():
        print("%s\t%s" % (k, v))



if __name__ == "__main__":
    main()
