//
// Created by phili on 28.07.2019.
//

#pragma once

#include <Placeholder.hpp>
#include <Parameter.hpp>
#include <Operation.hpp>
#include <iostream>

class Graph {
public:

	void updateParameters(HyperParameters& params);
	void computeForward();
	void computeBackwards();
	bool writeVariables(std::string dir);
	bool readVariables(std::string dir);



public:
	const std::shared_ptr<Placeholder> &getInput() const;

	void setInput(const std::shared_ptr<Placeholder> &input);

	const std::shared_ptr<Placeholder> &getLabels() const;

	void setLabels(const std::shared_ptr<Placeholder> &labels);

	void addParameter(std::shared_ptr<Parameter> variable);

	void addOperation(std::shared_ptr<Operation> operation);

	void setHyperParameters(const HyperParameters &hyperParameters);

private:
	std::shared_ptr<Placeholder> _input, _labels;
	std::vector<std::shared_ptr<Parameter>> _parameters;
	std::vector<std::shared_ptr<Operation>> _operations;


};


