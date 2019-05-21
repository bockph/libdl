//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "MUL.hpp"

/*MUL::MUL(std::shared_ptr<Node> X, std::shared_ptr<Node> W)
	:Operation(X,W)
{
	setInputA(X);
	setInputB(W);
//	_forward(X->getForward().rows(),W->getForward().cols());


}*/
void MUL::forwards(){

//	float tmp =1;
//
//	for(int i=0;i<getInputNodes().size();i++){
//		_forwardCache(i)=getInputNodes().at(i)->getForwardData();
//		tmp*= getInputNodes().at(i)->getForwardData();
//	}
//	setForwardData(tmp);
//	res = (A.cwiseProduct(B)).colwise().sum();

//	auto tmp =getInputA()->getForward().cwiseProduct(getInputB()->getForward()).colwise().sum();
/*
	std::cout<<"X"<<getInputA()->getForward().rows()<<","<<getInputA()->getForward().cols()<<std::endl;
	std::cout<<"W"<<getInputB()->getForward().rows()<<","<<getInputB()->getForward().cols()<<std::endl;
*/
	setForward(getInputA()->getForward()*getInputB()->getForward());
	//this results in a Vector containing in each row the result for a different input of the Batch

};
void MUL::backwards(float previousGradient) {
//	_gradients(0) = previousGradient.dot grad.dot(B.T), A.T.dot(grad)

	Eigen::MatrixXf inputGradient = getCurrentGradients()*(getInputB()->getForward().transpose());
	Eigen::MatrixXf weightGradient = (getInputA()->getForward().transpose())*getCurrentGradients();
	getInputA()->setCurrentGradients(inputGradient);

//	int size =getInputA()->getForward().rows();
//	for(int i =0;i<size;i++){
//		_gradients(i)=previousGradient;
//		for(int s =0;s<size;s++){
//			if(s!=i){
//				_gradients(i)*=_forwardCache(s);
//			}
//		}
//	}
}
void MUL::backwards() {

//if(first){
//	Eigen::MatrixXf inputGradient = 1*(getInputB()->getForward().transpose());
//	Eigen::MatrixXf weightGradient = (getInputA()->getForward().transpose())*1;
//	getInputA()->setCurrentGradients(inputGradient);
//	getInputB()->setCurrentGradients(weightGradient);
//}else{
/*
	std::cout<<"current Gradients"<<getCurrentGradients()<<std::endl;
	std::cout<<"Weights:"<<getInputA()->getForward()<<std::endl;
*/
	Eigen::MatrixXf inputGradient = getCurrentGradients()*(getInputB()->getForward().transpose());
	Eigen::MatrixXf weightGradient = (getInputA()->getForward().transpose())*getCurrentGradients();
	getInputA()->setCurrentGradients(inputGradient);
	getInputB()->setCurrentGradients(weightGradient);
//}



}