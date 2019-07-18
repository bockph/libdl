//
// Created by phili on 08.05.2019.
//

#include <memory>
#include "Node.hpp"

Node::Node():_outputChannels(0){}





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


const Eigen::MatrixXf &Node::getCurrentGradients() const {
	return currentGradients;
}

void Node::setCurrentGradients(const Eigen::MatrixXf &currentGradients) {
    if(currentGradients.rows()!=getForward().rows()&&currentGradients.cols()!=getForward().cols())
        throw::std::invalid_argument("The Gradient should have the same dimensions as the input");
	Node::currentGradients = currentGradients;
}

int Node::getOutputChannels() const {
    return _outputChannels;
}


int Node::getInputChannels() const {
    return _inputChannels;
}

void Node::setInputChannels(int inputChannels) {
    _inputChannels = inputChannels;
}

void Node::setOutputChannels(int outputChannels) {
    _outputChannels = outputChannels;
}



