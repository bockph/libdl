//
// Created by pbo on 18.06.19.
//

#include "ReLu.hpp"


#include <iostream>


void ReLu::forwards() {
    startTimeMeasurement();

    setForward(getInputA()->getForward().cwiseMax(0));

    stopTimeMeasurement(0);

};

float ReLu::deriveReLu(const float element) {
    if (element < 0)return 0;
    else return 1;
}

void ReLu::backwards() {
    startTimeMeasurement();

    std::function<float(float)> deriveReLu_WRAP = deriveReLu;
    Eigen::MatrixXf dReLu = getForward().unaryExpr(deriveReLu_WRAP);
    getInputA()->setCurrentGradients(getCurrentGradients().cwiseProduct(dReLu));

    stopTimeMeasurement(1);
}
