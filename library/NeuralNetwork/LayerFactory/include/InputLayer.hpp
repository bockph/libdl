//
// Created by pbo on 23.07.19.
//


#pragma once

#include <AbstractLayer.hpp>

/*!
 * Creates an InputLayer
 * this is the only Layer that does not take an input Layer as it is the starting Point
 */
class InputLayer : public AbstractLayer {
public:
    /*!
     *
     * @param computeGraph
     * @param batchSize
     * @param dataPoints amount of data one sample has
     * @param channel channels of the input
     */
    InputLayer(std::shared_ptr<Graph> computeGraph, int batchSize, int dataPoints, int channel);

    ~InputLayer() = default; //!default destructor
    /*!
     * During Training or Prediction this method should be used to set a batch for calculation
     * @param newMiniBatch the batch to be predicted, to be trained on
     */
    void updateX(Matrix &newMiniBatch);


};



