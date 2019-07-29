//
// Created by phili on 27.07.2019.
//

#pragma once

#include <Operation.hpp>


class ActivationFunction : public Operation {
public:
	ActivationFunction(std::shared_ptr<Node> X)
			: Operation(X,X->getOutputChannels()) {};

	~ActivationFunction() = default;


};


