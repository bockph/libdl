//
// Created by phili on 17.05.2019.
//

#include <iostream>
#include "MSEOp.hpp"

MSEOp::MSEOp(std::shared_ptr<Node> X, std::shared_ptr<Placeholder> labels)
        : LossFunction(std::move(X), labels) {};

void MSEOp::forwardPass() {

    auto diff = getInput()->getForward() - getLabels()->getForward();
    Matrix squared = diff.array().pow(2);

    Matrix mse(squared.rows(), squared.cols());
    mse.setZero();
    float tmp = 0;

    for (int j = 0; j < squared.cols(); j++) {
        for (int i = 0; i < squared.rows(); i++) {
            tmp += squared(i, j);
        }
        tmp /= squared.rows();
    }
    for (int i = 0; i < squared.rows(); i++) {
        for (int j = 0; j < squared.cols(); j++) {
            mse(i, j) = tmp;

        }
    }
    setForward(mse);
}


void MSEOp::backwardPass() {
    getInput()->setPreviousGradients(2 * (getForward() - getLabels()->getForward()));
}

