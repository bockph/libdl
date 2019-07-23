//
// Created by phili on 17.05.2019.
//

#pragma once


#include <Operation.hpp>

class MSEOp : public Operation {
public:
	MSEOp(std::shared_ptr<Node> X, std::shared_ptr<Node> C)
			: Operation(X, C) {};

	~MSEOp() = default;

	void forwards() override;

	void backwards() override;


};
