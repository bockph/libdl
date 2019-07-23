//
// Created by phili on 16.05.2019.
//

#include <iostream>
#include "SigmoidOP.hpp"


void SigmoidOP::forwards() {
    startTimeMeasurement();

    setForward(getInputA()->getForward().unaryExpr(std::ref(sigmoid)));

    stopTimeMeasurement(0);

};

void SigmoidOP::backwards() {
    startTimeMeasurement();

    auto tmp = getForward();
    tmp.setOnes();
    auto dSigmoid = getForward().cwiseProduct(tmp - getForward());
    getInputA()->setCurrentGradients(getCurrentGradients().cwiseProduct(dSigmoid));

    stopTimeMeasurement(1);
    /*
     * Debug Information
     */
    /* std::cout<<"Sigmoid FOrward:"<<getForward()<<std::endl;
     std::cout<<"Sigmoid Backwards:"<<getCurrentGradients()<<std::endl;*/
}

float SigmoidOP::sigmoid(float a) {
    return 1 / (1 + std::exp(-a));
}
