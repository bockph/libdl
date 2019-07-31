//
// Created by pbo on 23.07.19.
//


#pragma once

#include <AbstractLayer.hpp>

class LogitsLayer: public AbstractLayer {
public:
    LogitsLayer(std::shared_ptr<AbstractLayer> input,std::shared_ptr<Graph> computeGraph,int outputClasses);
    ~LogitsLayer()=default;


};


