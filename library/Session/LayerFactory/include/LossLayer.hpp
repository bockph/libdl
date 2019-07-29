//
// Created by pbo on 23.07.19.
//


#pragma once

#include <LossFunction.hpp>
#include <AbstractLayer.hpp>

class LossLayer: public AbstractLayer {
public:
    LossLayer(std::shared_ptr<AbstractLayer> input,std::shared_ptr<Graph> computeGraph, LossType losstype);
    ~LossLayer()=default;

    float getLoss();
    void updateLabels(Matrix newLabels);

};



