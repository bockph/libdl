//
// Created by pbo on 23.07.19.
//

#include <LossLayer.hpp>
#include <Placeholder.hpp>
#include <CrossEntropyOp.hpp>
#include <MSEOp.hpp>


LossLayer::LossLayer(std::shared_ptr<AbstractLayer> input, LossType losstype):
        AbstractLayer(input){



    /*
     * Initialization Labels
     */
    Matrix tmp;
   auto labels = std::make_shared<Placeholder>(tmp,0,0);


    /*
     * Initialization of Loss Operation
     */
    std::shared_ptr<Node> lossOp;

    switch (losstype){
        case LossType::CrossEntropy:
            lossOp = std::make_shared<CrossEntropyOp>(getInputNode(),labels);
            break;
        case LossType::MSE:
            lossOp = std::make_shared<MSEOp>(getInputNode(),labels);
            break;
        default:
            throw std::runtime_error(std::string("the selected LossType has yet not been Implemented in LossLayer class"));

    }

    setOutputNode(lossOp);

}
float LossLayer::getLoss(){

    return getOutputNode()->getForward()(0,0);
}
void LossLayer::updateLabels(Matrix newLabels){

    getOutputNode()->getInputB()->setForward(newLabels);
}

