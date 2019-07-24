//
// Created by pbo on 22.07.19.
//

#pragma once

#include <AbstractLayer.hpp>

class ConvolutionLayer: public AbstractLayer {
public:
    ConvolutionLayer(std::shared_ptr<AbstractLayer> input, ActivationFunction activationFunction,int amountFiltersint, int kernelDim, int stride=1, InitializationType initializationType=InitializationType::Xavier);
    ~ConvolutionLayer()=default;

    Matrix getFilterMatrix();
    Matrix getBiasMatrix();

    void setFilterMatrix(Matrix filter);
    void setBiasMatrix(Matrix bias);
private:


    std::shared_ptr<Weight> _filter;
    std::shared_ptr<Bias> _bias;

};


