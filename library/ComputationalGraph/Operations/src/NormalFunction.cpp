//
// Created by phili on 27.07.2019.
//

#include "NormalFunction.hpp"

NormalFunction::NormalFunction(std::shared_ptr<Node> X, std::shared_ptr<Parameter> parameter, int outputChannels)
		: Operation(X, outputChannels), _parameter(parameter) {}

NormalFunction::NormalFunction(std::shared_ptr<Node> X, int outputChannels)
		: Operation(X, outputChannels) {}

const std::shared_ptr<Parameter> &NormalFunction::getParameter() const {
	return _parameter;
}
