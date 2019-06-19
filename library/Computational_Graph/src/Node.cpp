//
// Created by phili on 08.05.2019.
//

#include <memory>
#include "Node.hpp"

void Node::addOutputNode(std::shared_ptr<Node> n){
	_outputNodes.push_back(n);
}

float Node::getForwardData() const {
	return _forwardData;
}

void Node::setForwardData(float forwardData) {
	_forwardData = forwardData;
}

float Node::getBackwardData() const {
	return _backwardData;
}

void Node::setBackwardData(float backwardData) {
	_backwardData = backwardData;
}

const Eigen::MatrixXf &Node::getForward() const {
	return _forward;
}

void Node::setForward(const Eigen::MatrixXf &forward) {
	_forward = forward;
}

const std::shared_ptr<Node> &Node::getInputA() const {
	return inputA;
}

void Node::setInputA(const std::shared_ptr<Node> &inputA) {
	Node::inputA = inputA;
}

const std::shared_ptr<Node> &Node::getInputB() const {
	return inputB;
}

void Node::setInputB(const std::shared_ptr<Node> &inputB) {
	Node::inputB = inputB;
}

float Node::getCurrentGradient() const {
	return currentGradient;
}

void Node::setCurrentGradient(float currentGradient) {
	Node::currentGradient = currentGradient;
}

const Eigen::MatrixXf &Node::getCurrentGradients() const {
	return currentGradients;
}

void Node::setCurrentGradients(const Eigen::MatrixXf &currentGradients) {
	Node::currentGradients = currentGradients;
}

int Node::getChannels() const {
    return _channels;
}

int Node::getOutputDim() const {
    return _outputDim;
}

void Node::setChannels(int channels) {
    _channels = channels;
}

void Node::setOutputDim(int outputDim) {
    _outputDim = outputDim;
}

int Node::getInputDim() const {
    return _inputDim;
}

void Node::setInputDim(int inputDim) {
    _inputDim = inputDim;
}


