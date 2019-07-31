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
     *  stores a node Containing the weights for the multiplication
     * @param X
     * @param numberClasses
     */
	SoftmaxOp(std::shared_ptr<Node> X,int numberClasses);

	~SoftmaxOp() override = default;//! default destructor

	/*!
	 *
	 */
	void forwardPass() override;

    /*!
     *  passes the gradient calculated in CrossEntropyLoss further backwards to the input Node X
     */
	void backwardPass() override;

private:
	int _numberClasses;
};
