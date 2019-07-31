//
// Created by phili on 30.06.2019.
//

#include "SoftmaxOp.hpp"


#include <utility>

SoftmaxOp::SoftmaxOp(std::shared_ptr<Node> X, int numberClasses)
		: ActivationFunction(std::move(X)), _numberClasses(numberClasses) {}

void SoftmaxOp::forwardPass() {
	Matrix output = getInput()->getForward();
    output.setZero();

	for (int i = 0; i < getInput()->getForward().rows(); i++) {
		Matrix result = getInput()->getForward().row(i);

		//adds numerical stability https://deepnotes.io/softmax-crossentropy, http://cs231n.github.io/linear-classify/#softmax
		result = result.array() - result.maxCoeff() + 0.0000000001;
		result = Eigen::exp(result.array());

		float tmpResult = result.sum();

		result = result.array() / tmpResult;

		result = result.array() + 0.0000000000000000000000000000000001;

        output.row(i) = result;
	}
	setForward(output);
}

void SoftmaxOp::backwardPass() {
	//Right Now Softmax does only work together with CrossEntropyOp, as the gradient of both together is calculated there
	//here we simply pass the gradients further backwards
	getInput()->setPreviousGradients(getPreviousGradients());

}

