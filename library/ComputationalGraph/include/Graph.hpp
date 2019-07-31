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
     * Adds an parameter to the list of parameters
     * @param variable
     */
    void addParameter(std::shared_ptr<Parameter> parameters);
    /*!
     * Adds an parameter to the list of parameters
     * ! the input Node of the passed operation must already be in _operations List or stored as label. Otherwise an exception is thrown!
     * @param operation
     */
    void addOperation(std::shared_ptr<Operation> operation);
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
	 * Writes all Parameters to path dir
	 * @param dir path/to/Parameter/files/Networkname
	 * @return success
	 */
	bool writeParameters(std::string dir);
    /*!
     * Reads all Parameters from path dir
     * @param dir path/to/Parameter/files/Networkname
     * @return success
     */
	bool readParameters(std::string dir);



private:
	std::shared_ptr<Placeholder> _input; //! represents the Input Node of the graph
    std::shared_ptr<Placeholder> _labels; //! represents the correct labels ( onlyneeded for a Loss function)
	std::vector<std::shared_ptr<Parameter>> _parameters; //!contains all updatable nodes of the graph
	std::vector<std::shared_ptr<Operation>> _operations; //!contains all operations of the graph ordered according to their position in the graph

    /*
     * Getters & Setters
     */

public:
    const std::shared_ptr<Placeholder> &getInput() const;

    void setInput(const std::shared_ptr<Placeholder> &input);

    const std::shared_ptr<Placeholder> &getLabels() const;

    void setLabels(const std::shared_ptr<Placeholder> &labels);




};


