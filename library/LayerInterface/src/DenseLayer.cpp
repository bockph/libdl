//
// Created by pbo on 23.07.19.
//

#include <DataInitialization.hpp>
#include <Weight.hpp>
#include <Bias.hpp>
#include <SummationOp.hpp>
#include <MultiplicationOp.hpp>
#include <SigmoidOP.hpp>
#include <ReLuOp.hpp>
#include <Weight.hpp>
#include "DenseLayer.hpp"


DenseLayer::DenseLayer(std::shared_ptr<AbstractLayer> input, ActivationFunction activationFunction,int amountNeurons,InitializationType initializationType):
        AbstractLayer(input),_activationFunction(activationFunction),_amountNeurons(amountNeurons),_initializationType(initializationType){



    /*
     * Initialization of Matrices
     */
    Matrix weightMatrix;
    switch (initializationType){
        case InitializationType ::Xavier:
            weightMatrix = DataInitialization::generateRandomMatrix(0,.1,input->getOutputSize(),_amountNeurons);

    }

    Matrix biasMatrix =Matrix::Zero(getBatchSize(),_amountNeurons);


    /*
     * Initialization of Operation Nodes
     */

    _weights = std::make_shared<Weight>(weightMatrix,1,1);
    _bias = std::make_shared<Bias>(biasMatrix);

    auto multiplicationOp = std::make_shared<MultiplicationOp>(getInputNode(),_weights);
    auto biasOp = std::make_shared<SummationOp>(multiplicationOp, _bias);

    std::shared_ptr<Node> activationOp;

    /*
     * Initialize activation Function
     */
    switch (_activationFunction){
        case ActivationFunction::ReLu:
            activationOp = std::make_shared<ReLuOp>(biasOp);
            break;
        case ActivationFunction::Sigmoid:
            activationOp = std::make_shared<SigmoidOP>(biasOp);
            break;
        case ActivationFunction ::None:
            activationOp=biasOp;
            break;
        default:
            throw std::runtime_error(std::string("the selected AcrivationFunction has yet not been Implemented in DenseLayer class"));
    }

    setOutputNode(activationOp);
    setOutputSize(amountNeurons);
    setOutputChannels(1);

}

Matrix DenseLayer::getWeightMatrix(){
    return _weights->getForward();
}
Matrix DenseLayer::getBiasMatrix(){
    return _bias->getForward();
}

void DenseLayer::setWeightMatrix(Matrix filter){
    _weights->setForward(filter);
}
void DenseLayer::setBiasMatrix(Matrix bias){
    _bias->setForward(bias);
}