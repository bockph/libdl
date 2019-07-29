//
// Created by phili on 28.07.2019.
//

#pragma once

#include <Graph.hpp>
//include normal Functions
#include <ConvolutionOp.hpp>
#include <SummationOp.hpp>
#include <MultiplicationOp.hpp>
#include <MaxPoolOp.hpp>
//include LossFunctions
#include <MSEOp.hpp>
#include <CrossEntropyOp.hpp>
//include ActivationFunctions
#include <ReLuOp.hpp>
#include <SigmoidOP.hpp>
#include <SoftmaxOp.hpp>

class OperationsFactory {
public:
	/*
	 * Normal Functions
	 */
	static std::shared_ptr<ConvolutionOp>
	createConvolutionOp(std::shared_ptr<Graph> graph, const std::shared_ptr<Node> input, Matrix filterMatrix,
						int channels, int stride);

	static std::shared_ptr<SummationOp>
	createSummationOp(std::shared_ptr<Graph> graph, const std::shared_ptr<Node> input, Matrix biasMatrix,
					  int channel);

	static std::shared_ptr<MultiplicationOp>
	createMultiplicationOp(std::shared_ptr<Graph> graph, const std::shared_ptr<Node> input, Matrix weightMatrix);

	static std::shared_ptr<MaxPoolOp>
	createMaxpoolOp(std::shared_ptr<Graph> graph, const std::shared_ptr<Node> input, int kernelDim, int stride);

	/*
	 * LossFunctions
	 */
	static std::shared_ptr<MSEOp>
	createMSEOp(std::shared_ptr<Graph> graph, const std::shared_ptr<Node> input, Matrix labelMatrix);

	static std::shared_ptr<CrossEntropyOp>
	createCrossEntropyOp(std::shared_ptr<Graph> graph, const std::shared_ptr<Node> input, Matrix labelMatrix);

	/*
	 * Activation Functions
	 */
	static std::shared_ptr<ReLuOp> createReLuOp(std::shared_ptr<Graph> graph, const std::shared_ptr<Node> input);

	static std::shared_ptr<SigmoidOP>
	createSigmoidOp(std::shared_ptr<Graph> graph, const std::shared_ptr<Node> input);

	static std::shared_ptr<SoftmaxOp>
	createSoftmaxOp(std::shared_ptr<Graph> graph, const std::shared_ptr<Node> input, int amountClasses);


};


