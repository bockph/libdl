//
// Created by pbo on 22.07.19.
//

#include <DataInitialization.hpp>
#include <SummationOp.hpp>
#include <SigmoidOP.hpp>
#include <ReLuOp.hpp>
#include <ConvolveFilterIM2COL.hpp>

#include "ConvolutionLayer.hpp"

ConvolutionLayer::ConvolutionLayer(std::shared_ptr<AbstractLayer> input, ActivationFunction activationFunction,int amountFilter, int kernelDim, int stride,InitializationType initializationType):
AbstractLayer(input){

    int inputSizeOneChannel=input->getOutputSize()/getInputChannels();
    int inputDim = std::sqrt(inputSizeOneChannel);
    int outputDim =std::floor((inputDim - kernelDim) / stride) + 1;
    int outputSize = std::pow(outputDim,2)*amountFilter;



    /*
     * Initialization of Matrices
     */
    Matrix filterMatrix;
    switch (initializationType){
        case InitializationType ::Xavier:
             filterMatrix = DataInitialization::generateRandomMatrix(0,.1,amountFilter,std::pow(kernelDim,2)*getInputChannels());
             break;
        default:
            throw std::runtime_error(std::string("the selected Initializationtype has yet not been Implemented in ConvolutionLayer class"));


    }

    Matrix biasMatrix =Matrix::Zero(getBatchSize(),outputDim*outputDim*amountFilter);


    /*
     * Initialization of Operation Nodes
     */

    _filter = std::make_shared<Variable>(filterMatrix , getInputChannels(), kernelDim);
    _bias = std::make_shared<Variable>(biasMatrix,amountFilter);

    std::shared_ptr<Node> convolutionOp = std::make_shared<ConvolveFilterIM2COL>(getInputNode(),_filter,stride);

    std::shared_ptr<SummationOp> biasOp = std::make_shared<SummationOp>(convolutionOp,_bias);

    std::shared_ptr<Node> activationOp;

    /*
     * Initialize activation Function
     */
    switch (activationFunction){
        case ActivationFunction::ReLu:
            activationOp = std::make_shared<ReLuOp>(biasOp);
            break;
        case ActivationFunction::Sigmoid:
            activationOp = std::make_shared<SigmoidOP>(biasOp);
            break;
        default:
            throw std::runtime_error(std::string("the selected AcrivationFunction has yet not been Implemented in ConvolutionLayer class"));
    }

    setOutputNode(activationOp);
    setOutputChannels(amountFilter);
    setOutputSize(outputSize);
}

Matrix ConvolutionLayer::getFilterMatrix(){
    return _filter->getForward();
}
Matrix ConvolutionLayer::getBiasMatrix(){
    return _bias->getForward();
}

void ConvolutionLayer::setFilterMatrix(Matrix filter){
    _filter->setForward(filter);
}
void ConvolutionLayer::setBiasMatrix(Matrix bias){
    _bias->setForward(bias);
}