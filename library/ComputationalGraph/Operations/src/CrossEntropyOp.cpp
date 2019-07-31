//
// Created by phili on 30.06.2019.
//

#include "CrossEntropyOp.hpp"

void CrossEntropyOp::forwardPass() {
	Matrix cwiseLogInput = Eigen::log(getInput()->getForward().array());
	Matrix cwiseMulWithLabels = cwiseLogInput.cwiseProduct(getLabels()->getForward());

	auto reduceSum = cwiseMulWithLabels.sum();
	float negative = reduceSum * -1;
	Matrix result(cwiseLogInput.rows(), cwiseLogInput.cols());

	for (int i = 0; i < result.rows(); i++) {
		for (int j = 0; j < result.cols(); j++) {
			result(i, j) = negative;
		}
	}
	setForward(result);
}


void CrossEntropyOp::backwardPass() {
	Matrix predictions =  getInput()->getForward();
	Matrix labels = getLabels()->getForward();
	Matrix tmp2 = predictions - labels;
	//TODO check if this can be deleted
	tmp2 = tmp2 / getInput()->getForward().rows();

	getInput()->setPreviousGradients(tmp2);


}

