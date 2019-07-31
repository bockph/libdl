//
// Created by pbo on 22.07.19.
//

#pragma once

#include <AbstractLayer.hpp>

/*!
 * Creates a Convolutional LAyer
 */
class ConvolutionLayer : public AbstractLayer {
public:
    /*!
     * Creates the layer
     *
     * @param input the previous Layer
     * @param computeGraph
     * @param activationFunction check commonDatatypes.h for possibilities
     * @param amountFilters
     * @param kernelDim
     * @param stride
     * @param initializationType check commonDatatypes.h for possibilities
     */
    ConvolutionLayer(std::shared_ptr<AbstractLayer> input, std::shared_ptr<Graph> computeGraph,
                     ActivationType activationFunction, int amountFilters, int kernelDim, int stride = 1,
                     InitializationType initializationType = InitializationType::Xavier);

    ~ConvolutionLayer() = default; //!default destructor


};


