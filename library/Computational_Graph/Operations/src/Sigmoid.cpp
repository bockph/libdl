//
// Created by phili on 16.05.2019.
//

#include <iostream>
#include "Sigmoid.hpp"

/*Sigmoid::Sigmoid(std::shared_ptr<Node> X)
		:Operation(X)
{
	setInputA(X);
//	_forward(X->getForward().rows(),W->getForward().cols());


}*/

void Sigmoid::forwards(){
//TODO: Might try using fast sigmoid f(x) = x / (1 + abs(x))
//https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm
	if(getInputNodes().size()!=1){
		throw std::invalid_argument("The Sigmoid function does only accept one Node");
	}else{
//		_forwardCache(0)=getInputNodes().at(0)->getForwardData();
//		setForwardData(sigmoid(_forwardCache(0)));

		setForward(getInputA()->getForward().unaryExpr(&sigmoid));
/*
		std::cout<<"Forward Sig:"<<getForward()<<std::endl;
*/

	}
};

void Sigmoid::backwards(float previousGradient) {

	float sigCalculated = sigmoid(_forwardCache(0));
	_gradients(0) = previousGradient*sigCalculated*(1-sigCalculated);

}
void Sigmoid::backwards() {
	auto tmp = getForward();
/*	std::cout<<"Gradients:"<<getCurrentGradients()<<std::endl;
	std::cout<<"Forward:"<<getForward()<<std::endl;*/
//	std::cout<<"C:"<<getInputB()->getForward()<<std::endl;
	tmp.setOnes();
//	if(first){
//		getInputA()->setCurrentGradients(1*getForward()*(tmp-getForward()));
//	}else{
auto tmp2 =getForward().cwiseProduct(tmp-getForward());
		getInputA()->setCurrentGradients(getCurrentGradients().cwiseProduct(tmp2));

//	}
}

float Sigmoid::sigmoid(float a) {
	return 1/(1+std::exp(-a));
}
