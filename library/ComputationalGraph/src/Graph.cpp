//
// Created by phili on 28.07.2019.
//

#include <IO.hpp>
#include "Graph.hpp"


void Graph::computeForward() {
	for (std::shared_ptr<Operation> op:_operations) {
		op->forwardPass();
	}
}

void Graph::computeBackwards() {
	Matrix gradientInit = _operations.back()->getForward();
	gradientInit.setOnes();
	_operations.back()->setPreviousGradients(gradientInit);
	for (std::vector<std::shared_ptr<Operation>>::reverse_iterator op = _operations.rbegin();
			op != _operations.rend(); ++op)
		(*op)->backwardPass();
}

void Graph::updateParameters(HyperParameters params) {
	//TODO check if gradient has been set or throw exception in Parameter
	for (std::shared_ptr<Parameter> parameter:_parameters)
		parameter->updateParameter(params);
}





bool Graph::writeParameters(std::string dir) {

	int idx = 0;
	for (std::shared_ptr<Parameter> parameter: _parameters) {
		if (!write_binary(dir + std::to_string(idx) + std::string(".bin"), parameter->getForward())) {
			return false;
		}
		idx++;
	}
	return true;
}

bool Graph::readParameters(std::string dir) {
	int idx = 0;
	for (std::shared_ptr<Parameter> parameter: _parameters) {


		Matrix tmpStore;
		if (!read_binary(dir + std::to_string(idx) + std::string(".bin"), tmpStore)) {
			return false;
		}
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

void Graph::addParameter(std::shared_ptr<Parameter> parameter) {
    if(parameter == nullptr)
        throw std::runtime_error("Can not add a nullptr to _parameters in graph");
	_parameters.push_back(parameter);
}

void Graph::addOperation(std::shared_ptr<Operation> operation) {
    if(operation == nullptr)
        throw std::runtime_error("Can not add a nullptr to _operations in graph");
    if(_operations.empty() )
        if(operation->getInput()!=_input)
            throw std::runtime_error("Cant add operation to graph, because the Input Node is not the label ");
    if(!_operations.empty()&&(operation->getInput())!=_operations.back())
        throw std::runtime_error("Cant add operation to graph, because the Input Node is not the last added operation");


    _operations.push_back(operation);

}


