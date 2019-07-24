//
// Created by pbo on 23.07.19.
//

#include <SoftmaxOp.hpp>
#include <LogitsLayer.hpp>


LogitsLayer::LogitsLayer(std::shared_ptr<AbstractLayer> input, int outputClasses) :
        AbstractLayer(input) {



    /*
     * Initialization of Softmax Node
     */

    auto softmax = std::make_shared<SoftmaxOp>(getInputNode(), outputClasses);


    setOutputNode(softmax);

}

