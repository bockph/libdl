//
// Created by pbo on 23.07.19.
//

#include <DataInitialization.hpp>
#include <Parameter.hpp>
#include <SummationOp.hpp>
#include <MultiplicationOp.hpp>
#include <SigmoidOP.hpp>
#include <ReLuOp.hpp>
#include <Parameter.hpp>
#include "DenseLayer.hpp"


DenseLayer::DenseLayer(std::shared_ptr<AbstractLayer> input, std::shared_ptr<Graph> computeGraph,
                       ActivationType activationFunction, int amountNeurons, InitializationType initializationType) :
        AbstractLayer(input, std::move(computeGraph)), _amountNeurons(amountNeurons),
        _initializationType(initializationType) {



    /*
     * Initialization of Matrices
     */
    Matrix weightMatrix;
    switch (initializationType) {
        case InitializationType::Xavier:
            weightMatrix = DataInitialization::generateRandomMatrix(input->getOutputSize(), _amountNeurons);
            break;
        default:
            throw std::runtime_error(std::string("the selected Initializationtype has yet not been Implemented in "
                                                 "DenseLayer class"));

    }

    Matrix biasMatrix = Matrix::Zero(getBatchSize(), _amountNeurons);

    /*
     * Initialization of Operation Nodes
     */
    auto multiplication = OperationsFactory::createMultiplicationOp(getComputeGraph(), getInputNode(), weightMatrix);
    auto biasSummation = OperationsFactory::createSummationOp(getComputeGraph(), multiplication, biasMatrix, 1);
    std::shared_ptr<Operation> activationOp;

    /*
     * Initialize activation Function
     */
    switch (activationFunction) {
        case ActivationType::ReLu:
            activationOp = OperationsFactory::createReLuOp(getComputeGraph(),
                                                           biasSummation);//std::make_shared<ReLuOp>             (biasOp);
            break;
        case ActivationType::Sigmoid:
            activationOp = OperationsFactory::createReLuOp(getComputeGraph(),
                                                           biasSummation);//      std::make_shared<SigmoidOP>(biasOp);
            break;
        case ActivationType::None:
            activationOp = biasSummation;
            break;
        default:
            throw std::runtime_error(
                    std::string("the selected AcrivationFunction has yet not been Implemented in DenseLayer class"));
    }

    setOutputNode(activationOp);
    setOutputSize(amountNeurons);
    setOutputChannels(1);

}

