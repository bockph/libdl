//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "MultiplicationOp.hpp"

void MultiplicationOp::forwards() {
    startTimeMeasurement();

    //this results in a Vector containing in each row the result for a different input of the Batch
    setForward(getInputA()->getForward() * getInputB()->getForward());

    stopTimeMeasurement(0);

};

void MultiplicationOp::backwards() {
    startTimeMeasurement();

    Eigen::MatrixXf dX = getCurrentGradients() * (getInputB()->getForward().transpose());
    Eigen::MatrixXf dW = (getInputA()->getForward().transpose()) * getCurrentGradients();

    getInputA()->setCurrentGradients(dX);
    getInputB()->setCurrentGradients(dW);
    stopTimeMeasurement(1);

    /*
     * Debug Information
     */
    /*int cols =getInputA()->getForward().cols();
    int rows =getInputB()->getForward().rows();*/
    /*int rows1 = getInputA()->getForward().rows();
    int cols1 = getInputA()->getForward().cols();

    int rows2 = getInputB()->getForward().rows();
    int cols2 = getInputB()->getForward().cols();
    std::cout<<"Mul FOrward:"<<getForward()<<std::endl;
    std::cout<<"Mul  Backwards:"<<getCurrentGradients()<<std::endl;*/

}
