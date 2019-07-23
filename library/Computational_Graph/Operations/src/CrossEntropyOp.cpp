//
// Created by phili on 30.06.2019.
//

#include <iostream>
#include "CrossEntropyOp.hpp"

void CrossEntropyOp::forwards() {
    startTimeMeasurement();
    Eigen::MatrixXf log = Eigen::log(getInputA()->getForward().array());
    Eigen::MatrixXf multiply = log.cwiseProduct(getInputB()->getForward());

    auto sumC = multiply.sum();
    float minus = sumC * -1;

    Eigen::MatrixXf result(log.rows(), log.cols());

    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.cols(); j++)
            result(i, j) = minus;
    }

    setForward(result);
    stopTimeMeasurement(0);

}


void CrossEntropyOp::backwards() {
    startTimeMeasurement();

    Eigen::MatrixXf c = getInputB()->getForward();
    Eigen::MatrixXf p = getInputA()->getForward();
    Eigen::MatrixXf tmp2 = p - c;
    tmp2 = tmp2 / getInputA()->getForward().rows();


    getInputA()->setCurrentGradients(tmp2);
    stopTimeMeasurement(1);

    /*
     * Debug Information
     */
    /*int rows1 = getInputA()->getForward().rows();
    int cols1 = getInputA()->getForward().cols();

    int rows2 = getInputB()->getForward().rows();
    int cols2 = getInputB()->getForward().cols();*/
    /*std::cout<<"CrossEntropy FOrward:"<<getForward()<<std::endl;
    std::cout<<"CrossEntropy Backwards:"<<getCurrentGradients()<<std::endl;*/

}

