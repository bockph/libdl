//
// Created by pbo on 22.07.19.
//

#include <DataInitialization.hpp>
#include <SummationOp.hpp>
#include <SigmoidOP.hpp>
#include <ReLuOp.hpp>
#include <ConvolutionOp.hpp>

#include "ConvolutionLayer.hpp"

ConvolutionLayer::ConvolutionLayer(std::shared_ptr<AbstractLayer> input, std::shared_ptr<Graph> computeGraph,
								   ActivationType activationFunction, int amountFilter, int kernelDim, int stride,
								   InitializationType initializationType)
		:
		AbstractLayer(input, computeGraph) {

	int inputSizeOneChannel = input->getOutputSize() / getInputChannels();
	int inputDim = static_cast<int>(std::sqrt(inputSizeOneChannel));
	int outputDim = static_cast<int>(std::floor((inputDim - kernelDim) / stride) + 1);
	int outputSize = static_cast<int>(std::pow(outputDim, 2) * amountFilter);



	/*
	 * Initialization of Matrices
	 */
	Matrix filterMatrix;
	switch (initializationType) {
		case InitializationType::Xavier:
			filterMatrix = DataInitialization::generateRandomMatrix(0, .1, amountFilter,
					static_cast<int>(std::pow(kernelDim, 2) * getInputChannels()));
			break;
		default:
			throw std::runtime_error(std::string("the selected Initializationtype has yet not been Implemented in ConvolutionLayer class"));


	}

	Matrix biasMatrix = Matrix::Zero(getBatchSize(), outputDim * outputDim * amountFilter);


	/*
	 * Initialization of Operation Nodes
	 */
	auto convolution = OperationsFactory::createConvolutionOp(getComputeGraph(), getInputNode(), filterMatrix, getInputChannels(), stride);
	auto biasSummation = OperationsFactory::createSummationOp(getComputeGraph(), convolution, biasMatrix, amountFilter);

	std::shared_ptr<Operation> activationOp;



	/*
	 * Initialize activation Function
	 */
	switch (activationFunction){
		case ActivationType::ReLu:
			activationOp = OperationsFactory::createReLuOp(getComputeGraph(), biasSummation);//std::make_shared<ReLuOp>             (biasOp);
			break;
		case ActivationType::Sigmoid:
			activationOp = OperationsFactory::createReLuOp(getComputeGraph(), biasSummation);//      std::make_shared<SigmoidOP>(biasOp);
			break;
		case ActivationType ::None:
			activationOp=biasSummation;
			break;
		default:
			throw std::runtime_error(std::string("the selected AcrivationFunction has yet not been Implemented in DenseLayer class"));
	}

	setOutputNode(activationOp);
	setOutputChannels(amountFilter);
	setOutputSize(outputSize);
}

Matrix ConvolutionLayer::getFilterMatrix() {
	return _filter->getForward();
}

Matrix ConvolutionLayer::getBiasMatrix() {
	return _bias->getForward();
}

void ConvolutionLayer::setFilterMatrix(Matrix filter) {
	_filter->setForward(filter);
}

void ConvolutionLayer::setBiasMatrix(Matrix bias) {
	_bias->setForward(bias);
}