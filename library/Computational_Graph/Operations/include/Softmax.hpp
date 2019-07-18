//
// Created by phili on 30.06.2019.
//


#pragma once


#include <Operation.hpp>

class Softmax : public Operation {
public:
	Softmax(std::shared_ptr<Node> X,int amountClasses)
			: Operation(X),_amountClasses(amountClasses) {};

	~Softmax() = default;

	void forwards() override;


	void backwards() override;

private:
	int _amountClasses;
};
