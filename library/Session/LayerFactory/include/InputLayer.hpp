//
// Created by pbo on 23.07.19.
//


#pragma once

#include <AbstractLayer.hpp>

class InputLayer: public AbstractLayer {
public:
    InputLayer(std::shared_ptr<Graph> computeGraph,int batchSize, int dataPoints, int channel);
    ~InputLayer()=default;
    void updateX(Matrix newMiniBatch);



};



