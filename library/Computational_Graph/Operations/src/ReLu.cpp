//
// Created by pbo on 18.06.19.
//

#include "ReLu.hpp"



#include <iostream>


void ReLu::forwards() {
    /*
 * GENERALL STUFF
 */
//    setChannels(getInputA()->getChannels());
    beforeForward();/*
 *
 */
    setForward(getInputA()->getForward().cwiseMax(0));

};
 float ReLu::deriveReLu (const float element) {
    if(element<0)return 0;
    else return 1;
}
void ReLu::backwards() {
    std::function<float(float)> deriveReLu_WRAP = deriveReLu;
    Eigen::MatrixXf dReLu = getForward().unaryExpr(deriveReLu_WRAP);
    getInputA()->setCurrentGradients(getCurrentGradients().cwiseProduct(dReLu));
}

std::string ReLu::printForward() {
    std::stringstream outStream;
    for (int i = 0; i < getForward().rows(); i++) {
        for (int j = 0; j < getForward().cols(); j++) {
            outStream << getForward()(i, j) << "\t";
        }
        outStream << std::endl;
    }
    return outStream.str();
}