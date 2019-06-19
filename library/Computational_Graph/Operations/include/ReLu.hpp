//
// Created by pbo on 18.06.19.
//

#pragma once

#include <Operation.hpp>

class ReLu : public Operation {
public:
    ReLu(std::shared_ptr<Node> X)
            : Operation(X) {};

    ~ReLu() = default;

    void forwards() override;

    void backwards() override;
    std::string printForward() override;

    static float deriveReLu(const float element) ;
};
