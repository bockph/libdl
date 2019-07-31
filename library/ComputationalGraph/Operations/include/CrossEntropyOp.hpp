//
// Created by phili on 30.06.2019.
//

#pragma once


#include "LossFunction.hpp"
/*!
 * Implements the CrossEntropy Loss
 */
class CrossEntropyOp : public LossFunction {
public:
    /*!
     * creates an LossFunction object
     * stores the labels which are need to calculate the loss
     * @param X softmax classified samples
     * @param labels groundTruth for initial samples
     */
	CrossEntropyOp(std::shared_ptr<Node> X, std::shared_ptr<Placeholder> labels);

	~CrossEntropyOp() = default;//! default destructor

	/*!
	 * calculates the cross-entropy loss and set the output as forward pass value
	 */
	void forwardPass() override;

	/*!
	 * calculates the softmax-crossEntropy-classifier gradient and passes it further to its input Node which must be a Softmax operation
     *  see also: https://deepnotes.io/softmax-crossentropy
	 */
	void backwardPass() override;


};

