import sys
projectDir='/mnt/c/Users/phili/Desktop/Projekte/libdl/'
# projectDir='/home/pbo/CLionProjects/libdl/'
sys.path.append(projectDir+'cmake-build-release/library/bindings')

import numpy as np
import libdl as dl

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def showPredictions(sampleBatch,labelBatch,network,i,dataStringPairs):
    prediction = network.predictBatch(sampleBatch,labelBatch)
    for (n,singlePrediction,label) in zip(enumerate(prediction),prediction,labelBatches[i]):
        maxP = max(singlePrediction)
        pred = np.where(singlePrediction==maxP)[0][0].astype(int)
        print("Actual:"+referenceList[dataStringPairs[i*batchSize+n[0]][1]][1])
        img1=mpimg.imread(dataStringPairs[i*batchSize+n[0]][0])
        #imgplot = plt.imshow(img)
        #plt.show()
        print('Predicted: '+referenceList[pred][1])
        print('Probability: '+maxP.astype(str))
        img2=mpimg.imread(projectDir+'data/'+referenceList[pred][2])
        # imgplot = plt.imshow(img)
        plt.show()

        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(img1)
        axarr[1].imshow(img2)