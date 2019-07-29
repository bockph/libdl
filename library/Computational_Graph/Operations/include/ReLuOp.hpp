//
// Created by pbo on 18.06.19.
//

#pragma once

#include "ActivationFunction.hpp"

class ReLuOp : public ActivationFunction {
public:
    ReLuOp(std::shared_ptr<Node> X)
            : ActivationFunction(X) {};

    ~ReLuOp() = default;

    void forwardPass() override;

    void backwardPass() override;

    static float deriveReLu(const float element) ;
};
