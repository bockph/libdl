//
// Created by phili on 08.05.2019.
//

#include <iostream>

#include "Operation.hpp"




Operation::Operation(std::shared_ptr<Node> X, int outputChannel):
_input(X),_inputChannels(X->getOutputChannels()){
	setOutputChannels(outputChannel);
}





void Operation::startTimeMeasurement() {
    _start = std::chrono::system_clock::now();

}

int Operation::stopTimeMeasurement(char function) {
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

    return elapsed_seconds;

}

int Operation::getForwardTime() const {
    return _forwardTime;
}

int Operation::getBackwardsTime() const {
    return _backwardsTime;
}

const std::shared_ptr<Node> &Operation::getInput() const {
	return _input;
}


int Operation::forwardPassWithMeasurement() {
	startTimeMeasurement();
	forwardPass();
	return stopTimeMeasurement(0);
}

int Operation::backwardPassWithMeasurement() {
	startTimeMeasurement();
	backwardPass();
	return stopTimeMeasurement(1);}


int Operation::getInputChannels() const {
	return _inputChannels;
}





