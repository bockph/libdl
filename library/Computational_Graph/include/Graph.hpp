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
	Graph(hyperParameters params=hyperParameters()):_hyperParameters(params){};
	~Graph()=default;
	void updateWeightsAndBiases();
	void predict();
	void train();
	bool writeVariables(std::string dir);
	bool readVariables(std::string dir);
private:
	void computeForward();
	void computeBackwards();

public:
	const std::shared_ptr<Placeholder> &getInput() const;

	void setInput(const std::shared_ptr<Placeholder> &input);

	const std::shared_ptr<Placeholder> &getLabels() const;

	void setLabels(const std::shared_ptr<Placeholder> &labels);

	void addVariable(std::shared_ptr<Parameter> variable);

	void addOperation(std::shared_ptr<Operation> operation);

	void setHyperParameters(const hyperParameters &hyperParameters);

private:
	std::shared_ptr<Placeholder> _input, _labels;
	std::vector<std::shared_ptr<Parameter>> _variables;
	std::vector<std::shared_ptr<Operation>> _operations;
	hyperParameters _hyperParameters;


};


