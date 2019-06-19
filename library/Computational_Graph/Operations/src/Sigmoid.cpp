//
// Created by phili on 16.05.2019.
//

#include <iostream>
#include "Sigmoid.hpp"


void Sigmoid::forwards() {
    /*
 * GENERALL STUFF
 */
//    setChannels(getInputA()->getChannels());
    beforeForward();/*
 *
 */
//TODO: Might try using fast sigmoid f(x) = x / (1 + abs(x))
//https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm
	setForward(getInputA()->getForward().unaryExpr(std::ref(sigmoid)));
};

void Sigmoid::backwards() {
	auto tmp = getForward();
	tmp.setOnes();
	auto dSigmoid = getForward().cwiseProduct(tmp - getForward());
	getInputA()->setCurrentGradients(getCurrentGradients().cwiseProduct(dSigmoid));

}

float Sigmoid::sigmoid(float a) {
	return 1 / (1 + std::exp(-a));
}
std::string Sigmoid::printForward() {
	return "Sigmoid:0";
}