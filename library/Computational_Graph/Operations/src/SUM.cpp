//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "SUM.hpp"

void SUM::forwards() {
    startTimeMeasurement();

/*
 * GENERALL STUFF
 */

//    setOutputChannels(getInputA()->getOutputChannels());
    beforeForward();
/*
 *
 */
    int rowsA = getInputA()->getForward().rows();
    int rowsB = getInputB()->getForward().rows();
    int colsA = getInputA()->getForward().cols();
    int colsB = getInputB()->getForward().cols();
//	std::cout<<"input:\n"<< getInputA()->getForward()<<std::endl;

	if (rowsA != rowsB) std::cout<<"!!!!"<<std::endl;
//		setForward(getInputA()->getForward() + getInputB()->getForward().replicate(rowsA, 1));
//	} else {

		setForward(getInputA()->getForward() + getInputB()->getForward());
//	}

    stopTimeMeasurement(0);

}

void SUM::backwards() {
    startTimeMeasurement();

    getInputB()->setCurrentGradients(getCurrentGradients());
	getInputA()->setCurrentGradients(getCurrentGradients());
    stopTimeMeasurement(1);

}

std::string SUM::printForward() {
	return "SUM:0";
}