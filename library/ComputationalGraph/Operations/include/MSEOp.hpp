//
// Created by phili on 17.05.2019.
//

#pragma once


#include "LossFunction.hpp"
/*!
 * Implements the Mean Square Error
 */
class MSEOp : public LossFunction {
public:
    /*!
     * creates an LossFunction object
     * stores the labels which are need to calculate the loss
     * @param X softmax classified samples
     * @param labels groundTruth for initial samples
     */
	MSEOp(std::shared_ptr<Node> X, std::shared_ptr<Placeholder> labels);

	~MSEOp() override = default;//! default destructor

    /*!
     * calculates the Mean Square Error of Input X with corresponding labels and sets the output as forward pass value
     */
	void forwardPass() override;

	/*!
	 * calculates the gradient w.r.t to X
	 */

	void backwardPass() override;


};
