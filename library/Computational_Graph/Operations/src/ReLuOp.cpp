//
// Created by pbo on 18.06.19.
//

#include "ReLuOp.hpp"


#include <iostream>


void ReLuOp::forwards() {


    startTimeMeasurement();

    setForward(getInputA()->getForward().cwiseMax(0));

    stopTimeMeasurement(0);

};

float ReLuOp::deriveReLu(const float element) {
    if (element < 0)return 0;
    else return 1;
}

void ReLuOp::backwards() {
    startTimeMeasurement();

    std::function<float(float)> deriveReLu_WRAP = deriveReLu;
    Eigen::MatrixXf dReLu = getForward().unaryExpr(deriveReLu_WRAP);
    getInputA()->setCurrentGradients(getCurrentGradients().cwiseProduct(dReLu));

    stopTimeMeasurement(1);

    /*
     * Debug INformation
     */
    /* int rows1 = getInputA()->getForward().rows();
     int cols1 = getInputA()->getForward().cols();
     std::cout<<"RELU FOrward:"<<getForward()<<std::endl;
     std::cout<<"RELU Backwards:"<<getCurrentGradients()<<std::endl;*/
}
