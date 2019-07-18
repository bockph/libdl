//
// Created by phili on 30.06.2019.
//

#include <iostream>
#include "CrossEntropyLoss.hpp"

void CrossEntropyLoss::forwards() {
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


void CrossEntropyLoss::backwards() {
    startTimeMeasurement();

    Eigen::MatrixXf c = getInputB()->getForward();
    Eigen::MatrixXf p = getInputA()->getForward();
    Eigen::MatrixXf tmp2 = p - c;
    tmp2 = tmp2 / getInputA()->getForward().rows();


    getInputA()->setCurrentGradients(tmp2);
    stopTimeMeasurement(1);

}

