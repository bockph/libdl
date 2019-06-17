//
// Created by pbo on 17.06.19.
//



#pragma once


#include <Operation.hpp>

class MaxPool : public Operation {
public:
    MaxPool(std::shared_ptr<Node> X,int windowSize, int stride =1)
            : Operation(X),_windowSize(windowSize),_stride(stride) {};

    ~MaxPool() = default;

    void forwards() override;

    void backwards() override;
    std::string printForward() override;
private:
    int _stride;
    int _windowSize;
};
