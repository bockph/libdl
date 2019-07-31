//
// Created by phili on 28.07.2019.
//

#pragma once

#include <Placeholder.hpp>
#include <Parameter.hpp>
#include <Operation.hpp>
#include <iostream>
/*!
 * This class controls the Behaviour of our list of Nodes aka Computational Graph.
 *
 */
class Graph {
public:
	/*!
	 * iterates through the list of Operations and executes forwardPass() on each.
	 * !this expects that each following Operation holds the previous Operation as input!
	 */
	void computeForward();

	/*!
	 * First sets the previousGradient for the last Operation to One.
	 * Second it iterates through the reversed list of Operations and  executes backwardsPass() on each.
	 * !this expects that each following Operation holds the previous Operation as input!
	 */
	void computeBackwards();

	/*!
	 * This function may first be called after execution of the Backward Pass
	 * It uses the gradients of computation w.r.t Parameter to update its forward pass value.
	 *
	 * @param params a HyperParameters object, specifying things like learningRate, BatchSize and which Optimizer
	 * should be used
	 */
	void updateParameters(HyperParameters params = HyperParameters());

	/*!
	 * Writes all Parameters 
	 * @param dir
	 * @return
	 */
	bool writeParameters(std::string dir);
	bool readParameters(std::string dir);



public:
	const std::shared_ptr<Placeholder> &getInput() const;

	void setInput(const std::shared_ptr<Placeholder> &input);

	const std::shared_ptr<Placeholder> &getLabels() const;

	void setLabels(const std::shared_ptr<Placeholder> &labels);

	void addParameter(std::shared_ptr<Parameter> variable);

	void addOperation(std::shared_ptr<Operation> operation);

private:
	std::shared_ptr<Placeholder> _input, _labels;
	std::vector<std::shared_ptr<Parameter>> _parameters;
	std::vector<std::shared_ptr<Operation>> _operations;


};


