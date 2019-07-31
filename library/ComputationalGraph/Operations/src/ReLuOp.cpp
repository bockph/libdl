//
// Created by pbo on 18.06.19.
//

#include "ReLuOp.hpp"


#include <iostream>

ReLuOp::ReLuOp(std::shared_ptr<Node> X)
		: ActivationFunction(X) {}

void ReLuOp::forwardPass() {
	setForward(getInput()->getForward().cwiseMax(0));
}

float ReLuOp::deriveReLu(const float element) {
	if (element < 0) { return 0; }
	else { return 1; }
}

void ReLuOp::backwardPass() {

	std::function<float(float)> deriveReLu_WRAP = deriveReLu;
	Eigen::MatrixXf dReLu = getForward().unaryExpr(deriveReLu_WRAP);
	getInput()->setPreviousGradients(getPreviousGradients().cwiseProduct(dReLu));


	/*
	 * Debug INformation
	 */
	/* int rows1 = getInputA()->getForward().rows();
	 int cols1 = getInputA()->getForward().cols();
	 std::cout<<"RELU FOrward:"<<getForward()<<std::endl;
	 std::cout<<"RELU Backwards:"<<getCurrentGradients()<<std::endl;*/
}
