//
// Created by phili on 08.05.2019.
//

#include <iostream>
#include "Operation.hpp"


Operation::Operation(std::shared_ptr<Node> X, std::shared_ptr<Node> W) {
	setInputA(X);
	setInputB(W);

	_inputNodes.push_back(X);
	_inputNodes.push_back(W);
	//add this Node as a Output Node for all reference inputNodes
	auto tmp(std::make_shared<Operation>(*this));
	X->addOutputNode(tmp);
	W->addOutputNode(tmp);

	auto ch = X->getChannels();

}

Operation::Operation(std::shared_ptr<Node> X) {
	_inputNodes.push_back(X);
	setInputA(X);
	//add this Node as a Output Node for all reference inputNodes
	auto tmp(std::make_shared<Operation>(*this));
	X->addOutputNode(tmp);

    auto ch = X->getChannels();


}

void Operation::beforeForward(){
    setChannels(getInputA()->getChannels());
    setOutputDim(getInputA()->getOutputDim());
}

const std::vector<std::shared_ptr<Node>> &Operation::getInputNodes() {
	return _inputNodes;
}
std::string Operation::printForward() {
	return "Operation:0";
}



