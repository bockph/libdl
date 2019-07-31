//
// Created by phili on 16.05.2019.
//

#include <iostream>
#include "SigmoidOP.hpp"

SigmoidOP::SigmoidOP(std::shared_ptr<Node> X)
		: ActivationFunction(std::move(X)) {}

void SigmoidOP::forwardPass() {
	setForward(getInput()->getForward().unaryExpr(std::ref(sigmoid)));
}

void SigmoidOP::backwardPass() {
	auto tmp = getForward();
	tmp.setOnes();
	auto dSigmoid = getForward().cwiseProduct(tmp - getForward());
	getInput()->setPreviousGradients(getPreviousGradients().cwiseProduct(dSigmoid));
}

float SigmoidOP::sigmoid(float a) {
	return 1 / (1 + std::exp(-a));
}
