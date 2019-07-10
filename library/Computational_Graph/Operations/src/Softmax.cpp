//
// Created by phili on 30.06.2019.
//

#include "Softmax.hpp"


#include <iostream>



void Softmax::forwards() {
    startTimeMeasurement();

    /*
 * GENERALL STUFF
 */
//    setOutputChannels(getInputA()->getOutputChannels());
	beforeForward();/*
 *
 */
    Eigen::MatrixXf tmp = getInputA()->getForward();

/*    std::cout<<"Output SoftBefore:\n"<< tmp<<std::endl;
//    std::cout<<"\nOUTPUT MAXPOOL:\n"<<test<<std::endl;
    std::cout<<"Output SoftBefore Max:\n"<< tmp.array().maxCoeff()<<std::endl;
    std::cout<<"Output SoftBefore Average:\n"<< tmp.mean()<<std::endl;

	std::cout<<"inputSo:\n"<<tmp<<std::endl;*/
	tmp.setZero();

	for(int i = 0; i<getInputA()->getForward().rows();i++){
		Eigen::MatrixXf result = getInputA()->getForward().block(i,0,1,tmp.cols());
//		std::cout<<"Start:\n"<<result<<std::endl;

		//adds numerical stability https://deepnotes.io/softmax-crossentropy, http://cs231n.github.io/linear-classify/#softmax
//		std::cout<<" MAX:\n"<<result.maxCoeff()<<std::endl;

		result=result.array() -result.maxCoeff()+0.0000000001;//* Eigen::MatrixXf::Ones(result.rows(),result.cols());
//		std::cout<<"Result-max:\n"<<result<<std::endl;

		result=Eigen::exp(result.array());
//		std::cout<<"EXP:\n"<<result<<std::endl;

		float tmpResult = result.sum();
//		std::cout<<"SUM:\n"<<tmpResult<<std::endl;

		result=result.array()/tmpResult;
//		std::cout<<"Result:\n"<<result<<std::endl;

		result=result.array()+0.0000000000000000000000000000000001;
//		std::cout<<"Result+:\n"<<result<<std::endl;

		tmp.block(i,0,1,tmp.cols())=result;
	}
		setForward(tmp);
//	std::cout<<"output:\n"<<getForward()<<std::endl;
    stopTimeMeasurement(0);


};
//TODO Check if this implementation of Softmax is really correct
void Softmax::backwards() {
    startTimeMeasurement();

    /*auto tmp = getForward();
    tmp.setOnes();
    auto dSoftMax = getForward().cwiseProduct(tmp - getForward());*/
//	getInputA()->setCurrentGradients(getCurrentGradients().cwiseProduct(dSoftMax));
	getInputA()->setCurrentGradients(getCurrentGradients());
//
//	std::cout<<"grads:\n"<<getInputA()->getCurrentGradients()<<std::endl;
//	std::cout<<"grads:\n"<<getCurrentGradients()<<std::endl;

    stopTimeMeasurement(1);

}

std::string Softmax::printForward() {
	return "Softmax:0";
}