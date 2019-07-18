//
// Created by phili on 17.05.2019.
//

#include <iostream>
#include "MSE.hpp"


void MSE::forwards() {
    /*
 * GENERALL STUFF
 */
//    setOutputChannels(getInputA()->getOutputChannels());
    beforeForward();/*
 *
 */
	auto diff = getInputA()->getForward() - getInputB()->getForward();
	Eigen::MatrixXf squared = diff.array().pow(2);

	Eigen::MatrixXf mse(squared.rows(), squared.cols());
	mse.setZero();
	float tmp = 0;

	for (int j = 0; j < squared.cols(); j++) {
		for (int i = 0; i < squared.rows(); i++) {
			tmp += squared(i, j);
		}
		tmp /= squared.rows();
	}
	for (int i = 0; i < squared.rows(); i++) {
		for(int j = 0;j<squared.cols();j++){
			mse(i, j) = tmp;

		}
	}
	setForward(mse);
}


void MSE::backwards() {
	getInputA()->setCurrentGradients(2 * (getForward() - getInputB()->getForward()));
}

