//
// Created by phili on 17.05.2019.
//

#pragma once


#include "LossFunction.hpp"

class MSEOp : public LossFunction {
public:
	MSEOp(std::shared_ptr<Node> X, std::shared_ptr<Placeholder> C)
			: LossFunction(X, C) {};

	~MSEOp() override = default;

	void forwardPass() override;

	void backwardPass() override;


};
