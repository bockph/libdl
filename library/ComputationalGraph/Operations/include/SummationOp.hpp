//
// Created by phili on 10.05.2019.
//
#pragma once

#include "NormalFunction.hpp"
/*!
 * Implements the Matrix summation that is mainly used for adding a Bias to a previous calculated forward pass of a Neuron/MultiplicationOP
 */
class SummationOp : public NormalFunction {
public:
    /*!
     *  creates NormalFunction object and passes the input outputChannesl through
     *  stores a node Containing the biases for the summation
     * @param X input Batch
     * @param biases the shape of biases should completly equal X
     */
	SummationOp(std::shared_ptr<Node> X, std::shared_ptr<Parameter> biases);

	~SummationOp() override = default;//!default destructor
    /*!
     * executes the matrix summation and stores the output as forward pass value
     */
	void forwardPass() override;

    /*!
     * calculates the gradients w.r.t to X and biases
     */
	void backwardPass() override;

};


