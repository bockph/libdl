//
// Created by phili on 08.05.2019.
//

#include <memory>
#include "Node.hpp"

void Node::addOutputNode(std::shared_ptr<Node> n){
	_outputNodes.push_back(n);
}

float Node::getDatavalue() const {
	return _datavalue;
}

void Node::setDatavalue(float datavalue) {
	_datavalue = datavalue;
}
