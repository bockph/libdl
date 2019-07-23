//
// Created by pbo on 23.07.19.
//


#pragma once

#include <AbstractLayer.hpp>

class InputLayer: public AbstractLayer {
public:
    InputLayer(Matrix miniBatch, int dim, int channel);
    ~InputLayer()=default;
    void updateX(Matrix newMiniBatch);
private:


    int _dim,_channel;

};



