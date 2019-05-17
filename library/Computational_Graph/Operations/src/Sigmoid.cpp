//
// Created by phili on 16.05.2019.
//

#include "Sigmoid.hpp"


void Sigmoid::forwards(){
//TODO: Might try using fast sigmoid f(x) = x / (1 + abs(x))
//https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm
	if(getInputNodes().size()!=1){
		throw std::invalid_argument("The Sigmoid function does only accept one Node");
	}else{
		setForwardData(sigmoid(getInputNodes().at(0)->getForwardData()));
		_forwardCache(0)=getInputNodes().at(0)->getForwardData();
	}
};
void Sigmoid::backwards(float previousGradient) {

	float sigCalculated = sigmoid(_forwardCache(0));
	_gradients(0) = previousGradient*sigCalculated*(1-sigCalculated);

}
float Sigmoid::sigmoid(float a) {
	return 1/(1+std::exp(-a));
}
