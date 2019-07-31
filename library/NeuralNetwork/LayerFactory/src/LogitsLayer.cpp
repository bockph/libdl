//
// Created by pbo on 23.07.19.
//

#include <SoftmaxOp.hpp>
#include <LogitsLayer.hpp>


LogitsLayer::LogitsLayer(std::shared_ptr<AbstractLayer> input,std::shared_ptr<Graph> computeGraph, int outputClasses) :
        AbstractLayer(input,computeGraph) {



    /*
     * Initialization of Softmax Node
     */

    auto softmax = OperationsFactory::createSoftmaxOp(getComputeGraph(),getInputNode(),outputClasses);


    setOutputNode(softmax);

}

