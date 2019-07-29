//
// Created by phili on 08.05.2019.
//

#include "Node.hpp"

const Eigen::MatrixXf &Node::getForward() const {
	return _forward;
}

void Node::setForward(const Eigen::MatrixXf &forward) {
	_forward = forward;
}

const Eigen::MatrixXf &Node::getPreviousGradients() const {
	return _previousGradients;
}

void Node::setPreviousGradients(const Eigen::MatrixXf &currentGradients) {
    if(currentGradients.rows()!=getForward().rows()&&currentGradients.cols()!=getForward().cols())
        throw::std::invalid_argument("The Gradient should have the same dimensions as the input");
	Node::_previousGradients = currentGradients;
}

int Node::getOutputChannels() const {
    return _outputChannels;
}

void Node::setOutputChannels(int outputChannels) {
    _outputChannels = outputChannels;
}



