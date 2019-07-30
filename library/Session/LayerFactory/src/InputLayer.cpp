//
// Created by pbo on 23.07.19.
//

#include <Placeholder.hpp>
#include "InputLayer.hpp"


InputLayer::InputLayer(std::shared_ptr<Graph> computeGraph,int batchSize, int dataPoints, int channel ) :
        AbstractLayer(computeGraph) {

    /*
     * Initialization of Placeholder
     */
    Matrix tmp;
    std::shared_ptr<Placeholder> X = std::make_shared<Placeholder>(tmp,channel);
	getComputeGraph()->setInput(X);

    setOutputNode(X);
    setOutputChannels(channel);
    //TODO check if OutputSize really needs to be set
    setOutputSize(dataPoints);
    setBatchSize(batchSize);

}

void InputLayer::updateX(Matrix &newMiniBatch){
    getOutputNode()->setForward(newMiniBatch);

}
