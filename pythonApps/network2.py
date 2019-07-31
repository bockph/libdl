import numpy as np
import sys
projectDir='/mnt/c/Users/phili/Desktop/Projekte/libdl/'
# projectDir='/home/pbo/CLionProjects/libdl/'
sys.path.append(projectDir+'cmake-build-release/library/bindings')


import libdl as dl

import matplotlib.pyplot as plt

nameNetwork = "multipleConvolutions"
nameNewNetwork = "batch16_Amount_100"
batchSize =16
epochs =15

learningRate=0.0001
amountBatches=100
dataStringPairs = dl.LegoDataLoader.shuffleData(projectDir+"data/")

data = dl.DataSet()
dl.LegoDataLoader.getData(batchSize*amountBatches,dataStringPairs,data)


graph = dl.Graph()

inputLayer = dl.InputLayer(graph,batchSize,160000,4)

convolution1 = dl.ConvolutionLayer(inputLayer,graph, dl.ActivationType.ReLu,32,8,2,dl.InitializationType.Xavier)
maxPool1 = dl.MaxPoolLayer(convolution1,graph,2,2)


convolution3 = dl.ConvolutionLayer(maxPool1,graph, dl.ActivationType.ReLu,64,5,2,dl.InitializationType.Xavier)

maxPool2 = dl.MaxPoolLayer(convolution3,graph,2,2)


dense1 = dl.DenseLayer(maxPool2,graph,dl.ActivationType.ReLu,1024,dl.InitializationType.Xavier)

dense2 = dl.DenseLayer(dense1,graph,dl.ActivationType.ReLu,16,dl.InitializationType.Xavier)

logits = dl.LogitsLayer(dense2,graph,16)

loss = dl.LossLayer(logits,graph, dl.LossType.CrossEntropy)


learningParameters = dl.HyperParameters(epochs,batchSize,learningRate)
network = dl.NeuralNetwork(graph,inputLayer,loss,learningParameters)
network.readVariables("/home/pbo/Schreibtisch/StoredValues/",nameNetwork)
trainingEvaluation = network.trainAndValidate(data,learningParameters,1)

network.writeVariables("/home/pbo/Schreibtisch/StoredValues/",nameNewNetwork)


epochs = np.arange(trainingEvaluation._hyperParameters._epochs)
# print(epoch)
# train_cost = losses[:,0]
# val_loss = losses[:,1]
# print(train_cost)
# print(val_loss)

# plt.subplot(2, 1, 1)
# plt.title('Training loss')
# plt.plot(epochs, train_cost)
# plt.plot(epochs, val_los)
# plt.xlabel('Iteration')
# plt.ylabel('Loss per iteration')
#
# plt.subplot(2, 1, 2)
# plt.title('Training loss for 10 batches')
# plt.plot(iterations10, train_cost10)
# plt.xlabel('Each 10 batch')
# plt.ylabel('Loss per each 10 batch')

plt.plot()
plt.title('Train vs Validation Loss')
plt.plot(epochs, trainingEvaluation._trainingLoss, '-o', label='train')
plt.plot(epochs, trainingEvaluation._validationLoss, '-o', label='val')
plt.plot(len(epochs), 'k--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy in percent')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.savefig("/home/pbo/Schreibtisch/StoredValues/loss_Graph_"+nameNewNetwork+".png")
plt.show()

plt.plot()
plt.title('Train vs Validation Accuracy')
plt.plot(epochs, trainingEvaluation._trainingAccuracy, '-o', label='train')
plt.plot(epochs, trainingEvaluation._validationAccuracy, '-o', label='val')
plt.plot(len(epochs), 'k--')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.savefig("/home/pbo/Schreibtisch/StoredValues/accuracy_Graph_"+nameNewNetwork+".png")
plt.show()






