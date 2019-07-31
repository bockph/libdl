//
// Created by pbo on 18.06.19.
//

#pragma once

#include "ActivationFunction.hpp"
/*!
 * Implements the ReLu activation function
 */
class ReLuOp : public ActivationFunction {
public:
    /*!
     * creates an ActivationFunction object using the input node X
     * @param X
     */
	explicit ReLuOp(std::shared_ptr<Node> X);

	~ReLuOp() override = default;//! default destructor

    /*!
	 * calculates ReLu for the whole input X and set the output value as forward pass value
	 */
	void forwardPass() override;

    /*!
     * calculates the sigmoid gradients w.r.t to input X using function deriveReLu
     */
	void backwardPass() override;

	/*!
	 * calculates the derivative of the ReLu function for a given element
	 * @param element
	 * @return dReLu(element)
	 */
	static float deriveReLu(float element);
};
