//
// Created by phili on 10.05.2019.
//

#include "SummationOp.hpp"

SummationOp::SummationOp(std::shared_ptr<Node> X, std::shared_ptr<Parameter> biases)
		: NormalFunction(X, std::move(biases), X->getOutputChannels()) {}

void SummationOp::forwardPass() {
	setForward(getInput()->getForward() + getParameter()->getForward());
}

void SummationOp::backwardPass() {
	getParameter()->setPreviousGradients(getPreviousGradients());
	getInput()->setPreviousGradients(getPreviousGradients());
}

