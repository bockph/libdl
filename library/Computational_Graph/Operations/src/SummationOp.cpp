//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "SummationOp.hpp"

void SummationOp::forwards() {


    startTimeMeasurement();
    setForward(getInputA()->getForward() + getInputB()->getForward());

    stopTimeMeasurement(0);

}

void SummationOp::backwards() {
    startTimeMeasurement();

    getInputB()->setCurrentGradients(getCurrentGradients());
    getInputA()->setCurrentGradients(getCurrentGradients());

    stopTimeMeasurement(1);


    /*
     * Debug Information
     */
    /*int rows1 =getInputA()->getForward().rows();
    int cols1 =getInputA()->getForward().cols();
    int rows2 =getInputB()->getForward().rows();
    int cols2 =getInputB()->getForward().cols();*/
   /* int rows1 = getInputA()->getForward().rows();
    int cols1 = getInputA()->getForward().cols();

    int rows2 = getInputB()->getForward().rows();
    int cols2 = getInputB()->getForward().cols();
    std::cout<<"SUM FOrward:"<<getForward()<<std::endl;
    std::cout<<"Sum Backwards:"<<getCurrentGradients()<<std::endl;*/
}

