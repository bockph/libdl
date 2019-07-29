//
// Created by phili on 16.05.2019.
//

#include <iostream>
#include "SigmoidOP.hpp"


void SigmoidOP::forwardPass() {

    setForward(getInput()->getForward().unaryExpr(std::ref(sigmoid)));


};

void SigmoidOP::backwardPass() {

    auto tmp = getForward();
    tmp.setOnes();
    auto dSigmoid = getForward().cwiseProduct(tmp - getForward());
	getInput()->setPreviousGradients(getPreviousGradients().cwiseProduct(dSigmoid));

    /*
     * Debug Information
     */
    /* std::cout<<"Sigmoid FOrward:"<<getForward()<<std::endl;
     std::cout<<"Sigmoid Backwards:"<<getCurrentGradients()<<std::endl;*/
}

float SigmoidOP::sigmoid(float a) {
    return 1 / (1 + std::exp(-a));
}
