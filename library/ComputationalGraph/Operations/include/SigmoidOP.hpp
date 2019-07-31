//
// Created by phili on 16.05.2019.
//
#pragma once


#include "ActivationFunction.hpp"

class SigmoidOP : public ActivationFunction {
public:
	explicit SigmoidOP(std::shared_ptr<Node> X);

	~SigmoidOP() override = default;

	void forwardPass() override;

	static float sigmoid(float a);

	void backwardPass() override;

};
