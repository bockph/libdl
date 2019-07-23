//
// Created by pbo on 23.07.19.
//

#include <Placeholder.hpp>
#include "InputLayer.hpp"


InputLayer::InputLayer(Matrix miniBatch, int dim, int channel) :
        AbstractLayer() {

    /*
     * Initialization of Placeholder
     */
    std::shared_ptr<Placeholder> X = std::make_shared<Placeholder>(miniBatch, dim, channel);


    setOutputNode(X);
    setOutputChannels(channel);
    setOutputSize(miniBatch.cols());
    setBatchSize(miniBatch.rows());

}

void InputLayer::updateX(Matrix newMiniBatch){
    getOutputNode()->setForward(newMiniBatch);
}
