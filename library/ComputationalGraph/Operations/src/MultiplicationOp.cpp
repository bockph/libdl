//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "MultiplicationOp.hpp"

void MultiplicationOp::forwardPass() {

    //this results in a Vector containing in each row the result for a different input of the Batch
    setForward(getInput()->getForward() * getParameter()->getForward());

};

void MultiplicationOp::backwardPass() {

    Eigen::MatrixXf dX = getPreviousGradients() * (getParameter()->getForward().transpose());
    Eigen::MatrixXf dW = (getInput()->getForward().transpose()) * getPreviousGradients();

	getInput()->setPreviousGradients(dX);
	getParameter()->setPreviousGradients(dW);

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
