//
// Created by phili on 27.07.2019.
//

#include "LossFunction.hpp"

LossFunction::LossFunction(std::shared_ptr<Node> X, std::shared_ptr<Placeholder> labels)
		: Operation(X, 1), _labels(labels) {}

const std::shared_ptr<Placeholder> &LossFunction::getLabels() const {
	return _labels;
}

const Matrix LossFunction::getPrediction() const {
	return getInput()->getForward();
}
