import numpy as np
import sys
# projectDir='/mnt/c/Users/phili/Desktop/Projekte/libdl/'
projectDir='/home/pbo/CLionProjects/libdl/'
dataStorage = projectDir+"data/Storage/"

sys.path.append(projectDir+'cmake-build-release/library/bindings')


import libdl as dl

import matplotlib.pyplot as plt

pretrainedNetwork = "batch16_Samples_10"
newNetworkName = "test"

#################################################
# specification of the most important Parameters#
#################################################
batchSize =16
epochs =5

learningRate=0.0001
amountBatches=5

#
#Loading of the Dataset
#
dataStringPairs = dl.LegoDataLoader.shuffleData(projectDir+"data/")
data = dl.DataSet()
dl.LegoDataLoader.getData(batchSize*amountBatches,dataStringPairs,data)

#constructing the compute Graph
graph = dl.Graph()

#
# Creation of the NN-Layers, syntax can be check in the corresponding classes in NeuralNetwork/LayerFactory/
#
inputLayer = dl.InputLayer(graph,batchSize,160000,4)

convolution1 = dl.ConvolutionLayer(inputLayer,graph, dl.ActivationType.ReLu,32,8,2,dl.InitializationType.Xavier)
maxPool1 = dl.MaxPoolLayer(convolution1,graph,2,2)


convolution3 = dl.ConvolutionLayer(maxPool1,graph, dl.ActivationType.ReLu,64,5,2,dl.InitializationType.Xavier)

maxPool2 = dl.MaxPoolLayer(convolution3,graph,2,2)


dense1 = dl.DenseLayer(maxPool2,graph,dl.ActivationType.ReLu,1024,dl.InitializationType.Xavier)

dense2 = dl.DenseLayer(dense1,graph,dl.ActivationType.ReLu,16,dl.InitializationType.Xavier)

logits = dl.LogitsLayer(dense2,graph,16)

loss = dl.LossLayer(logits,graph, dl.LossType.CrossEntropy)

#specifying hyper parameters, change the values in the top of the file
learningParameters = dl.HyperParameters(epochs,batchSize,learningRate)

# NeuralNetwork creation
network = dl.NeuralNetwork(graph,inputLayer,loss)

# for testing we do want to use the above specified pretrained Network, if not comment the follow line
network.readParameters(dataStorage,pretrainedNetwork)
trainingEvaluation = network.trainAndValidate(data,learningParameters,1)

#the following line can be uncommented if the weights should be written
#network.writeParameters(dataStorage,newNetworkName)



###
# Plotting the Training Results
###
epochs = np.arange(trainingEvaluation._hyperParameters._epochs)


plt.plot()
plt.title('Training vs Validation Loss')
plt.plot(epochs, trainingEvaluation._trainingLoss, '-o', label='train')
plt.plot(epochs, trainingEvaluation._validationLoss, '-o', label='val')
plt.plot(len(epochs), 'k--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.savefig(dataStorage+"loss_Graph_"+newNetworkName+".png")
plt.show()

plt.plot()
plt.title('Training vs Validation Accuracy')
plt.plot(epochs, trainingEvaluation._trainingAccuracy, '-o', label='train')
plt.plot(epochs, trainingEvaluation._validationAccuracy, '-o', label='val')
plt.plot(len(epochs), 'k--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.savefig(dataStorage+"accuracy_Graph_"+newNetworkName+".png")
plt.show()






