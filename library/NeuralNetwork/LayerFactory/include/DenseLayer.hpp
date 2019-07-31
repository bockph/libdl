//
// Created by pbo on 23.07.19.
//


#pragma once

#include <AbstractLayer.hpp>

class DenseLayer: public AbstractLayer {
public:
    DenseLayer(std::shared_ptr<AbstractLayer> input,std::shared_ptr<Graph> computeGraph, ActivationType
    activationFunction,int amountNeurons, InitializationType initializationType=InitializationType::Xavier);
    ~DenseLayer()=default;

    Matrix getWeightMatrix();
    Matrix getBiasMatrix();

    void setWeightMatrix(Matrix filter);
    void setBiasMatrix(Matrix bias);

private:

    ActivationType _activationFunction;
    InitializationType _initializationType;
    int _amountNeurons;

    std::shared_ptr<Parameter> _weights;
    std::shared_ptr<Parameter> _bias;
};



