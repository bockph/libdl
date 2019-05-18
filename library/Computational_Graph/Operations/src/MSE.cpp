//
// Created by phili on 17.05.2019.
//

#include "MSE.hpp"


void MSE::forwards(){

	if(getInputNodes().size()!=2){
		throw std::invalid_argument("The MSE function does only accept two Nodes. First the forward pass from an "
									  "activation function. Second the actual value");
	}else{
		_forwardCache(0)=getInputNodes().at(0)->getForwardData();
		_forwardCache(1)=getInputNodes().at(1)->getForwardData();
		setForwardData(std::pow((_forwardCache(0)-_forwardCache(1)),2));
	}
};
void MSE::backwards(float previousGradient) {
	_gradients(0) = 2*(_forwardCache(0)-_forwardCache(1));
}

