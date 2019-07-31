//
// Created by pbo on 18.06.19.
//

#pragma once

#include "ActivationFunction.hpp"

class ReLuOp : public ActivationFunction {
public:
	explicit ReLuOp(std::shared_ptr<Node> X);

	~ReLuOp() override = default;

	void forwardPass() override;

	void backwardPass() override;

	static float deriveReLu(float element);
};
