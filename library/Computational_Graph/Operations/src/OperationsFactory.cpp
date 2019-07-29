//
// Created by phili on 28.07.2019.
//

#include "OperationsFactory.hpp"

std::shared_ptr<ConvolutionOp> OperationsFactory::createConvolutionOp(std::shared_ptr<Graph> graph,
																	  const std::shared_ptr<Node> input,
																	  Matrix filterMatrix, int channels,
																	  int stride) {
	std::shared_ptr<Parameter> filter = std::make_shared<Parameter>(filterMatrix, channels);
	graph->addVariable(filter);
	std::shared_ptr<ConvolutionOp> convolutionOp = std::make_shared<ConvolutionOp>(input, filter, stride);
	graph->addOperation(convolutionOp);
	return convolutionOp;
}

std::shared_ptr<SummationOp> OperationsFactory::createSummationOp(std::shared_ptr<Graph> graph,
																  const std::shared_ptr<Node> input, Matrix
																  biasMatrix, int channel) {

	auto bias = std::make_shared<Parameter>(biasMatrix, channel);
	graph->addVariable(bias);

	auto summation = std::make_shared<SummationOp>(input, bias);
	graph->addOperation(summation);
	return summation;
}

std::shared_ptr<MultiplicationOp> OperationsFactory::createMultiplicationOp(std::shared_ptr<Graph> graph,
																			const std::shared_ptr<Node> input,
																			Matrix weightMatrix) {
	auto weights = std::make_shared<Parameter>(weightMatrix);
	graph->addVariable(weights);
	auto multiplication = std::make_shared<MultiplicationOp>(input, weights);
	graph->addOperation(multiplication);
	return multiplication;
}

std::shared_ptr<MaxPoolOp> OperationsFactory::createMaxpoolOp(std::shared_ptr<Graph> graph,
															  const std::shared_ptr<Node> input, int kernelDim,
															  int stride) {
	auto maxPool = std::make_shared<MaxPoolOp>(input, kernelDim, stride);
	graph->addOperation(maxPool);
	return maxPool;
}

std::shared_ptr<MSEOp> OperationsFactory::createMSEOp(std::shared_ptr<Graph> graph, const std::shared_ptr<Node> input,
													  Matrix labelMatrix) {

	auto labels = std::make_shared<Placeholder>(labelMatrix);
	graph->setLabels(labels);
	auto mse = std::make_shared<MSEOp>(input, labels);
	graph->addOperation(mse);
	return mse;
}

std::shared_ptr<CrossEntropyOp> OperationsFactory::createCrossEntropyOp(std::shared_ptr<Graph> graph,
																		const std::shared_ptr<Node> input,
																		Matrix labelMatrix) {
	auto labels = std::make_shared<Placeholder>(labelMatrix);
	graph->setLabels(labels);
	auto crossEntropy = std::make_shared<CrossEntropyOp>(input, labels);
	graph->addOperation(crossEntropy);
	return crossEntropy;
}

std::shared_ptr<ReLuOp> OperationsFactory::createReLuOp(std::shared_ptr<Graph> graph,
														const std::shared_ptr<Node> input) {
	auto reLu = std::make_shared<ReLuOp>(input);
	graph->addOperation(reLu);
	return reLu;
}

std::shared_ptr<SigmoidOP> OperationsFactory::createSigmoidOp(std::shared_ptr<Graph> graph,
															  const std::shared_ptr<Node> input) {
	auto sigmoid = std::make_shared<SigmoidOP>(input);
	graph->addOperation(sigmoid);
	return sigmoid;
}

std::shared_ptr<SoftmaxOp> OperationsFactory::createSoftmaxOp(std::shared_ptr<Graph> graph,
															  const std::shared_ptr<Node> input,
															  int amountClasses) {
	auto softmax = std::make_shared<SoftmaxOp>(input, amountClasses);
	graph->addOperation(softmax);
	return softmax;
}