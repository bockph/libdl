//
// Created by pbo on 23.07.19.
//

#include <LossLayer.hpp>
#include <Placeholder.hpp>
#include <CrossEntropyOp.hpp>
#include <MSEOp.hpp>


LossLayer::LossLayer(std::shared_ptr<AbstractLayer> input, std::shared_ptr<Graph> computeGraph, LossType losstype)
		:
		AbstractLayer(input, computeGraph) {



	/*
	 * Initialization Labels
	 */
	Matrix tmp;
//   auto labels = std::make_shared<Placeholder>(tmp,0,0);


	/*
	 * Initialization of Loss Operation
	 */
	std::shared_ptr<LossFunction> lossOp;

	switch (losstype) {
		case LossType::CrossEntropy:
			lossOp = OperationsFactory::createCrossEntropyOp(getComputeGraph(), getInputNode(), tmp);
//			std::make_shared<CrossEntropyOp>(getInputNode(), labels);
			break;
		case LossType::MSE:
			lossOp = OperationsFactory::createMSEOp(getComputeGraph(), getInputNode(), tmp);
			//std::make_shared<MSEOp>(getInputNode(), labels);
			break;
		default:
			throw std::runtime_error(std::string("the selected LossType has yet not been Implemented in LossLayer class"));
	}

	setOutputNode(lossOp);

}

float LossLayer::getLoss() {

	return getOutputNode()->getForward()(0, 0);
}

void LossLayer::updateLabels(Matrix newLabels) {
	if (std::dynamic_pointer_cast<LossFunction>(getOutputNode()) != nullptr) {
		std::dynamic_pointer_cast<LossFunction>(getOutputNode())->getLabels()->setForward(newLabels);
	}
}

