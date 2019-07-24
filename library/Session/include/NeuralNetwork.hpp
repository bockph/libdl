//
// Created by pbo on 24.07.19.
//

#pragma once

#include <InputLayer.hpp>
#include <LossLayer.hpp>
#include <commonDatatypes.hpp>
#include <Session.hpp>

class NeuralNetwork {
public:
    NeuralNetwork(const std::shared_ptr<InputLayer> inputLayer,const std::shared_ptr<LossLayer> lossLayer,const hyperParameters params = hyperParameters());
    void run(Matrix& miniBatch, Matrix& labels);

    bool writeVariables(std::string dir,std::string networkName);

    bool readVariables(std::string dir,std::string networkName);
    const hyperParameters &getParams() const;
    void setParams(const hyperParameters &params);
    float getLoss();
private:
    Session _computeGraph;
    std::shared_ptr<InputLayer> _inputLayer;
    std::shared_ptr<LossLayer> _lossLayer;
    bool _runAchieved;

};


