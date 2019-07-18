//
// Created by phili on 16.05.2019.
//
#pragma once


#include <Operation.hpp>

class Sigmoid : public Operation {
public:
	Sigmoid(std::shared_ptr<Node> X)
			: Operation(X) {};

	~Sigmoid() = default;

	void forwards() override;

	static float sigmoid(float a);

	void backwards() override;

};
