//
// Created by phili on 21.05.2019.
//


#include <iostream>
#include "Bias.hpp"

Bias::Bias(Eigen::MatrixXf m, int channel) {
	setForward(m);
	setOutputChannels(channel);

}


void Bias::backwards() {
	int rowsG = getCurrentGradients().rows();
	int rowsCurrent = getForward().rows();
	//TODO: Implement Learning Rate
	if (rowsCurrent != rowsG) {
		setForward(getForward().replicate(rowsG, 1) - 1 * getCurrentGradients());
	} else {
		setForward(getForward() - 1 * getCurrentGradients());
	}

}