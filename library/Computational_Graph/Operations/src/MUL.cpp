//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "MUL.hpp"

void MUL::forwards() {
    //General Stuff for Operations
    beforeForward();
    startTimeMeasurement();

    //this results in a Vector containing in each row the result for a different input of the Batch
    setForward(getInputA()->getForward() * getInputB()->getForward());

    stopTimeMeasurement(0);

};

void MUL::backwards() {
    startTimeMeasurement();

//    Eigen::MatrixXf dX = getCurrentGradients() * (getInputB()->getForward().transpose());
//    Eigen::MatrixXf dW = (getInputA()->getForward().transpose()) * getCurrentGradients();

    getInputA()->setCurrentGradients(getCurrentGradients() * (getInputB()->getForward().transpose()));
    getInputB()->setCurrentGradients((getInputA()->getForward().transpose()) * getCurrentGradients());
    stopTimeMeasurement(1);

}
