//
// Created by pbo on 18.06.19.
//

#pragma once

#include <Operation.hpp>

class ReLuOp : public Operation {
public:
    ReLuOp(std::shared_ptr<Node> X)
            : Operation(X) {};

    ~ReLuOp() = default;

    void forwards() override;

    void backwards() override;

    static float deriveReLu(const float element) ;
};
