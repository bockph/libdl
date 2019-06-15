//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "SUM.hpp"

void SUM::forwards() {

	int rowsA = getInputA()->getForward().rows();
	int rowsB = getInputB()->getForward().rows();

	if (rowsA != rowsB) {
		setForward(getInputA()->getForward() + getInputB()->getForward().replicate(rowsA, 1));
	} else {
		setForward(getInputA()->getForward() + getInputB()->getForward());
	}
}

void SUM::backwards() {
	getInputB()->setCurrentGradients(getCurrentGradients());
	getInputA()->setCurrentGradients(getCurrentGradients());

}

std::string SUM::printForward() {
	return "SUM:0";
}