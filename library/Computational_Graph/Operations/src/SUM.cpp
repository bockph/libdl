//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "SUM.hpp"

void SUM::forwards() {
    //General Stuff for Operations
    beforeForward();

    startTimeMeasurement();

    setForward(getInputA()->getForward() + getInputB()->getForward());

    stopTimeMeasurement(0);

}

void SUM::backwards() {
    startTimeMeasurement();

    getInputB()->setCurrentGradients(getCurrentGradients());
    getInputA()->setCurrentGradients(getCurrentGradients());

    stopTimeMeasurement(1);
}

