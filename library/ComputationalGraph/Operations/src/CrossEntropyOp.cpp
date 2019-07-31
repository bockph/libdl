//
// Created by phili on 30.06.2019.
//

#include "CrossEntropyOp.hpp"

CrossEntropyOp::CrossEntropyOp(std::shared_ptr<Node> X, std::shared_ptr<Placeholder> labels)
        : LossFunction(std::move(X), std::move(labels)) {};

void CrossEntropyOp::forwardPass() {
    Matrix cwiseLogInput = Eigen::log(getInput()->getForward().array());
    Matrix cwiseMulWithLabels = cwiseLogInput.cwiseProduct(getLabels()->getForward());

    auto reduceSum = cwiseMulWithLabels.sum();
    float negative = reduceSum * -1;
    Matrix result(cwiseLogInput.rows(), cwiseLogInput.cols());

    for (Eigen::Index i = 0; i < result.rows(); i++) {
        for (Eigen::Index j = 0; j < result.cols(); j++) {
            result(i, j) = negative;
        }
    }
    setForward(result);
}


void CrossEntropyOp::backwardPass() {
    Matrix predictions = getInput()->getForward();
    Matrix labels = getLabels()->getForward();
    Matrix dSoftmaxCrossEntropy = predictions - labels;
    dSoftmaxCrossEntropy = dSoftmaxCrossEntropy / getInput()->getForward().rows();
    getInput()->setPreviousGradients(dSoftmaxCrossEntropy);
}

