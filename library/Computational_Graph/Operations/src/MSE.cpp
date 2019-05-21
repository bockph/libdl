//
// Created by phili on 17.05.2019.
//

#include <iostream>
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

	float loss=0;
/*	std::cout<<"Input A:"<<getInputA()->getForward()<<std::endl;
	std::cout<<"Input C:"<<getInputB()->getForward()<<std::endl;*/
	auto diff = getInputA()->getForward()-getInputB()->getForward();
/*
	std::cout<<"Diff:"<<diff<<std::endl;
*/

	Eigen::MatrixXf squared = diff.array().pow(2);
/*
	std::cout<<"squared:"<<squared<<std::endl;
*/

	Eigen::MatrixXf mse(squared.rows(),squared.cols());
	mse.setZero();
	float tmp=0;

	for(int j =0;j<squared.cols();j++){
		for(int i =0;i<squared.rows();i++){
			float tf =squared(i,j);
//			mse(0,j) +=tf;
			tmp+=tf;
		}
//		mse(0,j) /=squared.rows();
		tmp /=squared.rows();

	}
	for(int i=0;i<squared.rows();i++){
		mse(i,0)=tmp;
	}
/*
	std::cout<<"MSE2:"<<mse<<std::endl;
*/

	setForward(mse);
};
void MSE::backwards(float previousGradient) {
	_gradients(0) = 2*(_forwardCache(0)-_forwardCache(1));
}
void MSE::backwards(bool first) {/*
	std::cout<<"Forward:"<<getForward()<<std::endl;
	std::cout<<"C:"<<getInputB()->getForward()<<std::endl;*/
getInputA()->setCurrentGradients(2*(getForward()-getInputB()->getForward()));

}

