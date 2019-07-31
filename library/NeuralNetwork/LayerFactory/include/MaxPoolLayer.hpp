//
// Created by pbo on 22.07.19.
//

#pragma once

#include <AbstractLayer.hpp>
/*!
 * Creates a max pooling layer
 */
class MaxPoolLayer: public AbstractLayer {
public:
    /*!
     *
     * @param input
     * @param computeGraph
     * @param kernelDim
     * @param stride
     */
    MaxPoolLayer(const std::shared_ptr<AbstractLayer>& input,std::shared_ptr<Graph> computeGraph, int kernelDim, int stride);
    ~MaxPoolLayer()=default; //!default destructor
};


