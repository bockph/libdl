//
// Created by phili on 08.05.2019.
//

#include <iostream>
#include "Operation.hpp"


Operation::Operation(std::shared_ptr<Node> X, std::shared_ptr<Node> W):
_amountOfInputs(X->getForward().rows()),
_inputDimX(X->getOutputDim()),
_inputDimW(W->getOutputDim()){
	setInputA(X);
	setInputB(W);

	_inputNodes.push_back(X);
	_inputNodes.push_back(W);
	//add this Node as a Output Node for all reference inputNodes
	auto tmp(std::make_shared<Operation>(*this));
	X->addOutputNode(tmp);
	W->addOutputNode(tmp);

    if(getInputA()->getOutputChannels()!= getInputB()->getOutputChannels()){
    	auto AChannels =getInputA()->getOutputChannels();
    	auto BChannels =getInputB()->getOutputChannels();
		throw std::invalid_argument("Input X and W should have the same amount of Channels");

	}
    setInputChannels(getInputA()->getOutputChannels());
    setOutputChannels(getInputChannels());
    setOutputDim(X->getOutputDim());
}

Operation::Operation(std::shared_ptr<Node> X):
        _amountOfInputs(X->getForward().rows()),
        _inputDimX(X->getOutputDim()) {
	_inputNodes.push_back(X);
	setInputA(X);
	//add this Node as a Output Node for all reference inputNodes
	auto tmp(std::make_shared<Operation>(*this));
	X->addOutputNode(tmp);

    setInputChannels(getInputA()->getOutputChannels());
	setOutputChannels(getInputChannels());
	setOutputDim(X->getOutputDim());



}
//This can not be applied to all, this needs to be changed
void Operation::beforeForward(){

    setOutputChannels(getInputA()->getOutputChannels());
    setOutputDim(getInputA()->getOutputDim());

}

const std::vector<std::shared_ptr<Node>> &Operation::getInputNodes() {
	return _inputNodes;
}
std::string Operation::printForward() {
	return "Operation:0";
}

int Operation::getAmountOfInputs() const {
    return _amountOfInputs;
}

void Operation::setAmountOfInputs(int amountOfInputs) {
    _amountOfInputs = amountOfInputs;
}

int Operation::getInputDimX() const {
    return _inputDimX;
}

void Operation::setInputDimX(int inputDimX) {
    _inputDimX = inputDimX;
}

int Operation::getInputDimW() const {
    return _inputDimW;
}

void Operation::setInputDimW(int inputDimW) {
    _inputDimW = inputDimW;
}



