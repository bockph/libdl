//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "MultiplicationOp.hpp"

MultiplicationOp::MultiplicationOp(std::shared_ptr<Node> X, std::shared_ptr<Parameter> weights)
        : NormalFunction(X, weights, X->getOutputChannels()) {}

void MultiplicationOp::forwardPass() {
    setForward(getInput()->getForward() * getParameter()->getForward());
}

void MultiplicationOp::backwardPass() {

    Matrix dX = getPreviousGradients() * (getParameter()->getForward().transpose());
    Matrix dW = (getInput()->getForward().transpose()) * getPreviousGradients();

    getInput()->setPreviousGradients(dX);
    getParameter()->setPreviousGradients(dW);
}
