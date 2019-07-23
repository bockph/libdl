//
// Created by phili on 30.06.2019.
//

#pragma once


#include <Operation.hpp>

class CrossEntropyOp : public Operation {
public:
	CrossEntropyOp(std::shared_ptr<Node> X, std::shared_ptr<Node> C)
			: Operation(X, C) {};

	~CrossEntropyOp() = default;

	void forwards() override;

	void backwards() override;


};
