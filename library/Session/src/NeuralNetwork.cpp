//
// Created by pbo on 24.07.19.
//

#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(const std::shared_ptr<InputLayer> inputLayer, const std::shared_ptr<LossLayer> lossLayer,
                             const hyperParameters params) :
        _computeGraph(lossLayer->getOutputNode(), params), _inputLayer(inputLayer), _lossLayer(lossLayer),_runAchieved(false) {
}

void NeuralNetwork::run(Matrix &miniBatch, Matrix &labels) {
    _inputLayer->updateX(miniBatch);
    _lossLayer->updateLabels(labels);
    _computeGraph.run();
    _runAchieved=true;
}

bool NeuralNetwork::writeVariables(std::string dir, std::string networkName) {
    return _computeGraph.writeVariables(dir+networkName);
}

bool NeuralNetwork::readVariables(std::string dir, std::string networkName) {
    return _computeGraph.readVariables(dir+networkName);

}

const hyperParameters &NeuralNetwork::getParams() const {
    return _computeGraph.getParams();
}

void NeuralNetwork::setParams(const hyperParameters &params) {
    _computeGraph.setParams(params);

}

float NeuralNetwork::getLoss() {
    if(!_runAchieved)throw std::runtime_error("The Neural Network can't return a Loss, if no runs has been performed yet.");
    return _lossLayer->getLoss();
}
