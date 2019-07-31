//
// Created by phili on 10.05.2019.
//
#pragma once


#include "NormalFunction.hpp"

class MultiplicationOp : public NormalFunction {
public:
	MultiplicationOp(std::shared_ptr<Node> X, std::shared_ptr<Parameter> W)
			: NormalFunction(X, W,X->getOutputChannels()) {}

	~MultiplicationOp() override = default;

	void forwardPass() override;

	void backwardPass() override;

};
