//
// Created by phili on 16.05.2019.
//

#include <iostream>
#include "Sigmoid.hpp"


void Sigmoid::forwards() {


    startTimeMeasurement();

    setForward(getInputA()->getForward().unaryExpr(std::ref(sigmoid)));

    stopTimeMeasurement(0);

};

void Sigmoid::backwards() {
    startTimeMeasurement();

    auto tmp = getForward();
    tmp.setOnes();
    auto dSigmoid = getForward().cwiseProduct(tmp - getForward());
    getInputA()->setCurrentGradients(getCurrentGradients().cwiseProduct(dSigmoid));

    stopTimeMeasurement(1);

}

float Sigmoid::sigmoid(float a) {
    return 1 / (1 + std::exp(-a));
}
