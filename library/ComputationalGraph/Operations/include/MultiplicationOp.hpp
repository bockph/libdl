//
// Created by phili on 10.05.2019.
//
#pragma once


#include "NormalFunction.hpp"

/*!
 * Implements the Matrix multiplication that simulates Neurons
 */
class MultiplicationOp : public NormalFunction {
public:
    /*!
     *  creates NormalFunction object and passes the input outputChannesl through
     *  stores a node Containing the weights for the multiplication
     * @param X input Batch
     * @param weights the amount of cols of the forward pass corresponds to the number of Neurons, the amount of rows must equal the size of one sample of X
     */
    MultiplicationOp(std::shared_ptr<Node> X, std::shared_ptr<Parameter> weights);

    ~MultiplicationOp() override = default;

    /*!
     * executes the Matrix multiplication and stores the output as forward pass value
     */
    void forwardPass() override;

    /*!
     * calculates the gradients w.r.t to X and weights
     */
    void backwardPass() override;

};
