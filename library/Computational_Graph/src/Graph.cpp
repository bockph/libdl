//
// Created by phili on 28.07.2019.
//

#include <IO.hpp>
#include "Graph.hpp"

void Graph::updateWeightsAndBiases() {
	for (std::shared_ptr<Parameter> variable:_variables) {
		variable->updateVariable(_hyperParameters);
	}
}

void Graph::predict() {
	computeForward();
}

void Graph::train() {
	computeForward();
	computeBackwards();
	updateWeightsAndBiases();
}

void Graph::computeForward() {
	for (std::shared_ptr<Operation> op:_operations) {
		op->forwardPass();
	}

}

void Graph::computeBackwards() {
	Eigen::MatrixXf tmp = _operations.back()->getForward();
	tmp.setOnes();
	_operations.back()->setPreviousGradients(tmp);
	for (std::vector<std::shared_ptr<Operation>>::reverse_iterator op = _operations.rbegin();
			op != _operations.rend(); ++op) {
		(*op)->backwardPass();
	}
}

bool Graph::writeVariables(std::string dir) {

	int idx =0;
    for (std::shared_ptr<Parameter> parameter: _variables) {
            if(!write_binary(dir+std::to_string(idx)+std::string(".bin"), parameter->getForward()))
                return false;
        idx++;
    }
	return true;
}

bool Graph::readVariables(std::string dir) {
	    int idx =0;
    for (std::shared_ptr<Parameter> parameter: _variables) {


            Matrix tmpStore;
            if(!read_binary(dir+std::to_string(idx)+std::string(".bin"), tmpStore))
                return false;
            parameter->setForward(tmpStore);


        idx++;
    }

	return true;
}

const std::shared_ptr<Placeholder> &Graph::getInput() const {
	return _input;
}

void Graph::setInput(const std::shared_ptr<Placeholder> &input) {
	_input = input;
}

const std::shared_ptr<Placeholder> &Graph::getLabels() const {
	return _labels;
}

void Graph::setLabels(const std::shared_ptr<Placeholder> &labels) {
	_labels = labels;
}

void Graph::addVariable(std::shared_ptr<Parameter> variable) {
	_variables.push_back(variable);
}

void Graph::addOperation(std::shared_ptr<Operation> operation) {
	_operations.push_back(operation);

}


void Graph::setHyperParameters(const hyperParameters &hyperParameters) {
	Graph::_hyperParameters = hyperParameters;
}
