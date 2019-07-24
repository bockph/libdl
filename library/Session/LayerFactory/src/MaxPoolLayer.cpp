//
// Created by pbo on 22.07.19.
//


#include <MaxPoolOp.hpp>

#include "MaxPoolLayer.hpp"

MaxPoolLayer::MaxPoolLayer(std::shared_ptr<AbstractLayer> input, int kernelDim, int stride):
AbstractLayer(input){
    int inputSizeOneChannel=input->getOutputSize()/getInputChannels();
    int inputDim = std::sqrt(inputSizeOneChannel);
    int outputDim =std::floor((inputDim - kernelDim) / stride) + 1;
    int outputSize = std::pow(outputDim,2)*getInputChannels();
    /*
     * Initialization of Operation Nodes
     */
    auto maxPool = std::make_shared<MaxPoolOp>(getInputNode(),kernelDim,stride);

    setOutputNode(maxPool);
    setOutputChannels(input->getOutputChannels());
    setOutputSize(outputSize);
}

