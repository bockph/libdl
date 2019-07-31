//
// Created by phili on 30.06.2019.
//


#pragma once

#include "ActivationFunction.hpp"

/*!
 * Implements the Softmax Classifier
 * 	Right Now Softmax does only work in combination with CrossEntropyOp, as the gradient of both together is calculated there and then just passed backwards
 */
class SoftmaxOp : public ActivationFunction {
public:
    /*!
     * creates an ActivationFunction object
     * stores the number of classes that need to be predicted
     * @param X input to be transformed
     * @param numberClasses #predictionClasses
     */
    SoftmaxOp(std::shared_ptr<Node> X, int numberClasses);

    ~SoftmaxOp() override = default;//! default destructor

    /*!
     * performs softmax classification
     */
    void forwardPass() override;

    /*!
     *  passes the softmax-crossEntropy-classifier gradient calculated in CrossEntropyLoss further backwards to the input Node X
     *  see also: https://deepnotes.io/softmax-crossentropy
     */
    void backwardPass() override;

private:
    int _numberClasses;
};
