//
// Created by phili on 16.05.2019.
//
#pragma once


#include "ActivationFunction.hpp"
/*!
 * Implements the Sigmoid activation function
 */
class SigmoidOP : public ActivationFunction {
public:
    /*!
     * creates an ActivationFunction object using the input node X
     * @param X
     */
	explicit SigmoidOP(std::shared_ptr<Node> X);

	~SigmoidOP() override = default;//! default destructor

	/*!
	 * calculates sigmoid for the whole input X and set the output value as forward pass value
	 */
	void forwardPass() override;

	/*!
	 * calculates the sigmoid function
	 * @param a
	 * @return sigmoid(a)
	 */
	static float sigmoid(float a);

	/*!
	 * calculates the sigmoid gradients w.r.t to input X
	 */
	void backwardPass() override;

};
