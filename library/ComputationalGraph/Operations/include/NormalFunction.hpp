//
// Created by phili on 27.07.2019.
//

#pragma once

#include <Operation.hpp>
#include <Parameter.hpp>
/*!
 * A NormalFunction operation represents typical operations that take a Weight or Bias
 * Right now this also counts for the MaxPool Layer, however in future it should get its own Base class as MaxPool does not hold Weights or Biases
 */
class NormalFunction : public Operation {
public:
    /*!
     *  - creates the operation Object using the the obligatory Input Node and the outputChannels.
     *  - stores the parameters for this operation
     * @param X input Node
     * @param parameter parameters such as weights, biases, filters
     * @param outputChannels the expected outputChannels for this operation
     */
	NormalFunction(std::shared_ptr<Node> X, std::shared_ptr<Parameter> parameter, int outputChannels);
    /*!
     * - creates the operation Object using the the obligatory Input Node and the outputChannels.
     * @param X input Node
     * @param outputChannels the expected outputChannels for this operation
     */
	NormalFunction(std::shared_ptr<Node> X, int outputChannels);

	~NormalFunction() override = default;//!Default Destructor

private:
	std::shared_ptr<Parameter> _parameter; //!arameters such as weights, biases, filters

	/*
	 * Getters & Setters
	 */
public:
	const std::shared_ptr<Parameter> &getParameter() const;
};


