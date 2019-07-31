//
// Created by phili on 10.05.2019.
//
#pragma once

#include "NormalFunction.hpp"

class SummationOp : public NormalFunction {
public:
	SummationOp(std::shared_ptr<Node> X, std::shared_ptr<Parameter> W);

	~SummationOp() override = default;

	void forwardPass() override;

	void backwardPass() override;

};


