//
// Created by phili on 30.06.2019.
//


#pragma once


#include <Operation.hpp>

class SoftmaxOp : public Operation {
public:
	SoftmaxOp(std::shared_ptr<Node> X,int amountClasses)
			: Operation(X),_amountClasses(amountClasses) {};

	~SoftmaxOp() = default;

	void forwards() override;


	void backwards() override;

private:
	int _amountClasses;
};
