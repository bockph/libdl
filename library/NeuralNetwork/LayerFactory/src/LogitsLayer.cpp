//
// Created by pbo on 23.07.19.
//

#include <SoftmaxOp.hpp>
#include <LogitsLayer.hpp>
#include <utility>


LogitsLayer::LogitsLayer(std::shared_ptr<AbstractLayer> input,std::shared_ptr<Graph> computeGraph, int outputClasses) :
        AbstractLayer(std::move(input),computeGraph) {



    /*
     * Initialization of Softmax Node
     */

    auto softmax = OperationsFactory::createSoftmaxOp(getComputeGraph(),getInputNode(),outputClasses);


    setOutputNode(softmax);

}

