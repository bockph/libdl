//
// Created by pbo on 18.07.19.
//

#include "AbstractLayer.hpp"

AbstractLayer::AbstractLayer(std::shared_ptr<AbstractLayer> input):_inputLayer(input),
_inputNode(input->getOutputNode()),_inputChannels(input->getOutputChannels()),_batchSize(input->getBatchSize())
{}

const std::shared_ptr<Node> &AbstractLayer::getInputNode() const {
    return _inputNode;
}

const std::shared_ptr<Node> &AbstractLayer::getOutputNode() const {
    return _outputNode;
}

void AbstractLayer::setOutputNode(const std::shared_ptr<Node> &outputNode) {
    _outputNode = outputNode;
}

int AbstractLayer::getOutputChannels() const {
    return _outputChannels;
}

void AbstractLayer::setOutputChannels(int outputChannels) {
    AbstractLayer::_outputChannels = outputChannels;
}

int AbstractLayer::getInputChannels() const {
    return _inputChannels;
}



int AbstractLayer::getBatchSize() const {
    return _batchSize;
}

void AbstractLayer::setBatchSize(int batchSize) {
    _batchSize = batchSize;
}

int AbstractLayer::getOutputSize() const {
    return _outputSize;
}

void AbstractLayer::setOutputSize(int outputSize) {
    _outputSize = outputSize;
}
