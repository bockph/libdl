//
// Created by pbo on 23.07.19.
//

#include <Placeholder.hpp>
#include "InputLayer.hpp"


InputLayer::InputLayer(int batchSize, int dim, int channel) :
        AbstractLayer() {

    /*
     * Initialization of Placeholder
     */
    Matrix tmp;
    std::shared_ptr<Placeholder> X = std::make_shared<Placeholder>(tmp, dim, channel);


    setOutputNode(X);
    setOutputChannels(channel);
    setOutputSize(std::pow(dim,2)*channel);
    setBatchSize(batchSize);

}

void InputLayer::updateX(Matrix newMiniBatch){
    getOutputNode()->setForward(newMiniBatch);

}
