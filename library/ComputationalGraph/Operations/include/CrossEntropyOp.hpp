//
// Created by phili on 30.06.2019.
//

#pragma once


#include "LossFunction.hpp"

class CrossEntropyOp : public LossFunction {
public:
	CrossEntropyOp(std::shared_ptr<Node> X, std::shared_ptr<Placeholder> C)
			: LossFunction(X, C) {};

	~CrossEntropyOp() = default;

	void forwardPass() override;

	void backwardPass() override;


};
