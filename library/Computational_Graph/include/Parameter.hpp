//
// Created by phili on 14.06.2019.
//
#pragma once

#include <Node.hpp>
#include <commonDatatypes.hpp>

/*!
 * This Node holds the Parameters that define the function of the Computational Graph.
 * In terms of a Neural Networks, this class holds the functionality for Filters, Weights and Biases, which may be
 * updated during Backpropagation
 */
class Parameter : public Node {

public:
	/*!
	 * Constructs a Filter and sets the Output to the filterMatrix.
	 * Further it initializes the momentum and RMSprob for possible future Updates using the ADAM optimizer
	 *
	 * @param filter each Row corresponds to one Kernel, the size of one Kernel equals std::pow(KernelDim,2)*channel
	 * @param channel the Amount of Channels each Kernel has
	 */
	Parameter(Matrix &filter, int channel );

	/*!
	 * Constructs a Weight or Bias and sets the Output to the inputMatrix.
	 * Further it initializes the momentum and RMSprob for possible future Updates using the ADAM optimizer
	 * @param input
	 */
	Parameter(Matrix &input);

	/*!
	 * Default Destructor
	 */
    ~Parameter() = default;

    /*!
     * updates the output of the Variable according to the Optimizer and Hyperparameters that are passed.
     * @param hyperParameters Object containing all hyperParameters for Neural Networks
     */
	void updateVariable(const hyperParameters& hyperParameters);


private:
    Eigen::MatrixXf _s1; //! stores the RMSprop value for ADAM Optimizer
    Eigen::MatrixXf _v1; //! stores the momentum for ADAM Optimizer

};


