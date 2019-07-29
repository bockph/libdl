//
// Created by phili on 30.06.2019.
//

#include <iostream>
#include "CrossEntropyOp.hpp"

void CrossEntropyOp::forwardPass() {
    Eigen::MatrixXf log = Eigen::log(getInput()->getForward().array());
    Eigen::MatrixXf multiply = log.cwiseProduct(getLabels()->getForward());

    auto sumC = multiply.sum();
    float minus = sumC * -1;

    Eigen::MatrixXf result(log.rows(), log.cols());

    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.cols(); j++)
            result(i, j) = minus;
    }

    setForward(result);

}


void CrossEntropyOp::backwardPass() {
	int rows1 = getInput()->getForward().rows();
	int cols1 = getInput()->getForward().cols();

	int rows2 = getLabels()->getForward().rows();
	int cols2 = getLabels()->getForward().cols();
	Eigen::MatrixXf p = getInput()->getForward();

    Eigen::MatrixXf c = getLabels()->getForward();
    Eigen::MatrixXf tmp2 = p - c;
    tmp2 = tmp2 / getInput()->getForward().rows();


	getInput()->setPreviousGradients(tmp2);

    /*
     * Debug Information
     */
    /*int rows1 = getInput()->getForward().rows();
    int cols1 = getInput()->getForward().cols();

    int rows2 = getLabels()->getForward().rows();
    int cols2 = getLabels()->getForward().cols();*/
    /*std::cout<<"CrossEntropy FOrward:"<<getForward()<<std::endl;
    std::cout<<"CrossEntropy Backwards:"<<getCurrentGradients()<<std::endl;*/

}

