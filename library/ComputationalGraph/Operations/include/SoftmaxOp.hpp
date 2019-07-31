//
// Created by phili on 30.06.2019.
//


#pragma once


#include "ActivationFunction.hpp"

class SoftmaxOp : public ActivationFunction {
public:
	SoftmaxOp(std::shared_ptr<Node> X,int amountClasses);

	~SoftmaxOp() override = default;

	void forwardPass() override;


	void backwardPass() override;

private:
	int _amountClasses;
};
