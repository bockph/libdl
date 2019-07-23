//
// Created by phili on 10.05.2019.
//
#pragma once

#include <Operation.hpp>

class SummationOp : public Operation {
public:
	SummationOp(std::shared_ptr<Node> X, std::shared_ptr<Node> W)
			: Operation(X, W) {};

	~SummationOp() = default;

	void forwards() override;

	void backwards() override;

};


