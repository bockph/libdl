//
// Created by phili on 30.06.2019.
//

#include "Softmax.hpp"


#include <iostream>



void Softmax::forwards() {
	/*
 * GENERALL STUFF
 */
//    setOutputChannels(getInputA()->getOutputChannels());
	beforeForward();/*
 *
 */
	Eigen::MatrixXf result = getInputA()->getForward();
	//adds numerical stability https://deepnotes.io/softmax-crossentropy, http://cs231n.github.io/linear-classify/#softmax
	result -=result.maxCoeff()* Eigen::MatrixXf::Ones(result.rows(),result.cols());
	Eigen::exp(result.array());
	result/=result.sum();



	setForward(result);
};

void Softmax::backwards() {
	auto tmp = getForward();
	tmp.setOnes();
	auto dSoftMax = getForward().cwiseProduct(tmp - getForward());
	getInputA()->setCurrentGradients(getCurrentGradients().cwiseProduct(dSoftMax));

}

std::string Softmax::printForward() {
	return "Softmax:0";
}