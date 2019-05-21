//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "MUL.hpp"

void MUL::forwards() {
	//this results in a Vector containing in each row the result for a different input of the Batch
	setForward(getInputA()->getForward() * getInputB()->getForward());

};

void MUL::backwards() {

	Eigen::MatrixXf inputGradient = getCurrentGradients() * (getInputB()->getForward().transpose());
	Eigen::MatrixXf weightGradient = (getInputA()->getForward().transpose()) * getCurrentGradients();
	getInputA()->setCurrentGradients(inputGradient);
	getInputB()->setCurrentGradients(weightGradient);

}