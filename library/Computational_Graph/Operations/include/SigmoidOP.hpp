//
// Created by phili on 16.05.2019.
//
#pragma once


#include <Operation.hpp>

class SigmoidOP : public Operation {
public:
	SigmoidOP(std::shared_ptr<Node> X)
			: Operation(X) {};

	~SigmoidOP() = default;

	void forwards() override;

	static float sigmoid(float a);

	void backwards() override;

};
