//
// Created by pbo on 18.06.19.
//

#include "ReLu.hpp"



#include <iostream>


void ReLu::forwards() {
    startTimeMeasurement();

    /*
 * GENERALL STUFF
 */
//    setOutputChannels(getInputA()->getOutputChannels());
    beforeForward();/*
 *
 */
    setForward(getInputA()->getForward().cwiseMax(0));

//    Eigen::MatrixXf test = getForward();
//    test.transpose().eval();
//    test.resize(getOutputDim(),getOutputDim()*getOutputChannels());
  /*  std::cout<<"Output RELU:\n"<< getForward()<<std::endl;
//    std::cout<<"\nOUTPUT MAXPOOL:\n"<<test<<std::endl;
    std::cout<<"Output RLU Max:\n"<< getForward().array().maxCoeff()<<std::endl;
    std::cout<<"Output RELU Average:\n"<< getForward().mean()<<std::endl;
    int x=2+3;*/
    stopTimeMeasurement(0);

};
 float ReLu::deriveReLu (const float element) {
    if(element<0)return 0;
    else return 1;
}
void ReLu::backwards() {
    startTimeMeasurement();

    std::function<float(float)> deriveReLu_WRAP = deriveReLu;
    Eigen::MatrixXf dReLu = getForward().unaryExpr(deriveReLu_WRAP);
    getInputA()->setCurrentGradients(getCurrentGradients().cwiseProduct(dReLu));
    stopTimeMeasurement(1);

}
