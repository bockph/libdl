//
// Created by pbo on 17.06.19.
//



#pragma once


#include <Operation.hpp>

class MaxPoolOp : public Operation {
public:
    MaxPoolOp(std::shared_ptr<Node> X,int windowSize, int stride =1)
            : Operation(X),_windowSize(windowSize),_stride(stride) {};

    ~MaxPoolOp() = default;

    void forwards() override;

    void backwards() override;

    const Eigen::MatrixXf &getMaxIndexMatrix() const;

    void setMaxIndexMatrix(const Eigen::MatrixXf &maxIndexMatrix);

private:
    int _stride;
    int _windowSize;
    Eigen::MatrixXf _maxIndexMatrix;
};
