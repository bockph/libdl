//
// Created by phili on 17.05.2019.
//

#pragma once


#include <Operation.hpp>

class MSE : public Operation {
public:
	MSE(std::shared_ptr<Node> X, std::shared_ptr<Node> C)
			: Operation(X, C) {};

	~MSE() = default;

	void forwards() override;

	void backwards() override;
	std::string printForward() override;


};
