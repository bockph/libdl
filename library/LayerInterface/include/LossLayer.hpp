//
// Created by pbo on 23.07.19.
//


#pragma once

#include <AbstractLayer.hpp>

class LossLayer: public AbstractLayer {
public:
    LossLayer(std::shared_ptr<AbstractLayer> input, Matrix labelsMatrix,LossType losstype);
    ~LossLayer()=default;

    float getLoss();
    void updateLabels(Matrix newLabels);
};



