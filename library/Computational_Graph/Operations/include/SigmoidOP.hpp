//
// Created by phili on 16.05.2019.
//
#pragma once


#include "ActivationFunction.hpp"

class SigmoidOP : public ActivationFunction {
public:
	SigmoidOP(std::shared_ptr<Node> X)
			: ActivationFunction(X) {};

	~SigmoidOP() = default;

	void forwardPass() override;

	static float sigmoid(float a);

	void backwardPass() override;

};
