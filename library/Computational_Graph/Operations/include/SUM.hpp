//
// Created by phili on 10.05.2019.
//
#pragma once

#include <Operation.hpp>

class SUM : public Operation {
public:
	SUM(std::shared_ptr<Node> X, std::shared_ptr<Node> W)
			: Operation(X, W) {};

	~SUM() = default;

	void forwards() override;

	void backwards() override;

};


