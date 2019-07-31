//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "SummationOp.hpp"

void SummationOp::forwardPass() {
    setForward(getInput()->getForward() + getParameter()->getForward());


}

void SummationOp::backwardPass() {

	getParameter()->setPreviousGradients(getPreviousGradients());
	getInput()->setPreviousGradients(getPreviousGradients());



    /*
     * Debug Information
     */
    /*int rows1 =getInput()->getForward().rows();
    int cols1 =getInput()->getForward().cols();
    int rows2 =getVariable()->getForward().rows();
    int cols2 =getParameter()->getForward().cols();*/
   /* int rows1 = getInputA()->getForward().rows();
    int cols1 = getInputA()->getForward().cols();

    int rows2 = getInputB()->getForward().rows();
    int cols2 = getInputB()->getForward().cols();
    std::cout<<"SUM FOrward:"<<getForward()<<std::endl;
    std::cout<<"Sum Backwards:"<<getCurrentGradients()<<std::endl;*/
}

