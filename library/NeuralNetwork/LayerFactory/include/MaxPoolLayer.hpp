//
// Created by pbo on 22.07.19.
//

#pragma once

#include <AbstractLayer.hpp>

class MaxPoolLayer: public AbstractLayer {
public:
    MaxPoolLayer(std::shared_ptr<AbstractLayer> input,std::shared_ptr<Graph> computeGraph, int kernelDim, int stride);
    ~MaxPoolLayer()=default;



};


