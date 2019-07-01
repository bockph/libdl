//
// Created by phili on 30.06.2019.
//

#pragma once


#include <Operation.hpp>

class CrossEntropyLoss : public Operation {
public:
	CrossEntropyLoss(std::shared_ptr<Node> X, std::shared_ptr<Node> C)
			: Operation(X, C) {};

	~CrossEntropyLoss() = default;

	void forwards() override;

	void backwards() override;
	std::string printForward() override;


};
