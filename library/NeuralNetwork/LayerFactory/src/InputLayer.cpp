//
// Created by pbo on 23.07.19.
//

#include <Placeholder.hpp>
#include "InputLayer.hpp"


InputLayer::InputLayer(std::shared_ptr<Graph> computeGraph,int batchSize, int dataPoints, int channel ) :
        AbstractLayer(std::move(computeGraph)) {

    /*
     * Initialization of Placeholder
     */
    Matrix tmp;
    std::shared_ptr<Placeholder> X = std::make_shared<Placeholder>(tmp,channel);
    //We have no Factory for a Placeholder Node therefore the inptu must set manually
	getComputeGraph()->setInput(X);

    setOutputNode(X);
    setOutputChannels(channel);
    setOutputSize(dataPoints);
    setBatchSize(batchSize);

}

void InputLayer::updateX(Matrix &newMiniBatch){
    getOutputNode()->setForward(newMiniBatch);

}
