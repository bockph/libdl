//
// Created by pbo on 23.07.19.
//


#pragma once

#include <AbstractLayer.hpp>
/*!
 * Creates a Softmax Classification Layer
 */
class LogitsLayer: public AbstractLayer {
public:
    /*!
     * Creates a Logits Layer using the softmax classifier
     * @param input
     * @param computeGraph
     * @param outputClasses the number of labels
     */
    LogitsLayer(std::shared_ptr<AbstractLayer> input,std::shared_ptr<Graph> computeGraph,int outputClasses);
    ~LogitsLayer()=default;//! default destructor


};



