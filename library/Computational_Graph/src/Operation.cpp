//
// Created by phili on 08.05.2019.
//

#include <iostream>
#include "Operation.hpp"


Operation::Operation(std::shared_ptr<Node> X, std::shared_ptr<Node> W){
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
    	//TODO uncomment the exception
//		throw std::invalid_argument("Input X and W should have the same amount of Channels");

	}
    setInputChannels(getInputA()->getOutputChannels());
    setOutputChannels(getInputChannels());
}

Operation::Operation(std::shared_ptr<Node> X){
    setInputA(X);
	_inputNodes.push_back(X);

	//add this Node as a Output Node for all reference inputNodes
	auto tmp(std::make_shared<Operation>(*this));
	X->addOutputNode(tmp);
    setInputChannels(getInputA()->getOutputChannels());

	setOutputChannels(getInputChannels());



}
//This can not be applied to all, this needs to be changed
void Operation::beforeForward(){

//    setOutputChannels(getInputA()->getOutputChannels());

}

const std::vector<std::shared_ptr<Node>> &Operation::getInputNodes() {
	return _inputNodes;
}






void Operation::startTimeMeasurement() {
    _start = std::chrono::system_clock::now();

}

void Operation::stopTimeMeasurement(char function) {
    _end = std::chrono::system_clock::now();

    int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
            (_end-_start).count();

    switch (function){
        case 0: _forwardTime=elapsed_seconds;
            break;
        case 1: _backwardsTime=elapsed_seconds;
            break;
        default:break;
    }


}

int Operation::getForwardTime() const {
    return _forwardTime;
}

int Operation::getBackwardsTime() const {
    return _backwardsTime;
}



