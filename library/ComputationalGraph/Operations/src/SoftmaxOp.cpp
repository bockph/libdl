//
// Created by phili on 30.06.2019.
//

#include "SoftmaxOp.hpp"


#include <iostream>

SoftmaxOp::SoftmaxOp(std::shared_ptr<Node> X, int amountClasses)
		: ActivationFunction(X), _amountClasses(amountClasses) {}

void SoftmaxOp::forwardPass() {
	Eigen::MatrixXf tmp = getInput()->getForward();
	tmp.setZero();

	for (int i = 0; i < getInput()->getForward().rows(); i++) {
		Eigen::MatrixXf result = getInput()->getForward().block(i, 0, 1, tmp.cols());

		//adds numerical stability https://deepnotes.io/softmax-crossentropy, http://cs231n.github.io/linear-classify/#softmax
		result = result.array() - result.maxCoeff() + 0.0000000001;
		result = Eigen::exp(result.array());

		float tmpResult = result.sum();

		result = result.array() / tmpResult;

		result = result.array() + 0.0000000000000000000000000000000001;

		tmp.block(i, 0, 1, tmp.cols()) = result;
	}
	setForward(tmp);
}

void SoftmaxOp::backwardPass() {
	//Right Now Softmax does only work together with CrossEntropyOp, as the gradient of both together is calculated there and then just passed forward
	getInput()->setPreviousGradients(getPreviousGradients());

	/*
	 * Debug Information
	 */
	/*int rows1 = getInputA()->getForward().rows();
	int cols1 = getInputA()->getForward().cols();
	std::cout<<"Softmax FOrward:"<<getForward()<<std::endl;
	std::cout<<"Softmax Backwards:"<<getCurrentGradients()<<std::endl;*/

}

