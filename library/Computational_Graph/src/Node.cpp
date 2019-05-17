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


