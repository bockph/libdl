//
// Created by pbo on 18.07.19.
//
#pragma once

#include <Node.hpp>
#include <memory>
#include <DataInitialization.hpp>

class AbstractLayer {
public:
    enum ActivationFunction {
        ReLu, Sigmoid, LeakyRelu,None
    };
    enum InitializationType{
        Xavier
    };
    enum LossType{
        CrossEntropy,MSE
    };

    AbstractLayer(std::shared_ptr<AbstractLayer> input);
    AbstractLayer(){};

    ~AbstractLayer() = default;

    const std::shared_ptr<Node> &getInputNode() const;

    const std::shared_ptr<Node> &getOutputNode() const;

    void setOutputNode(const std::shared_ptr<Node> &outputNode);


    int getOutputChannels() const;

    void setOutputChannels(int outputChannels);

    int getInputChannels() const;

    int getOutputSize() const;

    void setOutputSize(int outputSize);

    int getBatchSize() const;

    void setBatchSize(int batchSize);

private:
    std::shared_ptr<AbstractLayer> _inputLayer;
    std::shared_ptr<Node> _inputNode;
    std::shared_ptr<Node> _outputNode;
    int _outputChannels;
    int _outputSize;
    int _inputChannels;
    int _batchSize;

};

