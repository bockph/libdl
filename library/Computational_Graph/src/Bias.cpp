//
// Created by phili on 21.05.2019.
//


#include <iostream>
#include "Bias.hpp"

Bias::Bias(Eigen::MatrixXf m) {
	setForward(m);
}


void Bias::backwards() {
	int rowsG = getCurrentGradients().rows();
	int rowsCurrent = getForward().rows();
	//TODO: Implement Learning Rate
	if (rowsCurrent != rowsG) {
		setForward(getForward().replicate(rowsG, 1) - 50 * getCurrentGradients());
	} else {
		setForward(getForward() - 50 * getCurrentGradients());
	}

}