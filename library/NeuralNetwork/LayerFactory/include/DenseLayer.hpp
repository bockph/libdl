//
// Created by pbo on 23.07.19.
//

#pragma once

#include <AbstractLayer.hpp>
/*!
 * Creates a DenseLayer
 */
class DenseLayer: public AbstractLayer {
public:
    /*!
     *  Create the Layer
     * @param input
     * @param computeGraph
     * @param activationFunction check commonDatatypes.h for possibilities
     * @param amountNeurons
     * @param initializationType check commonDatatypes.h for possibilities
     */
    DenseLayer(std::shared_ptr<AbstractLayer> input,std::shared_ptr<Graph> computeGraph, ActivationType
    activationFunction,int amountNeurons, InitializationType initializationType=InitializationType::Xavier);
    ~DenseLayer()=default; //!default destructor



};



