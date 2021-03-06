//
// Created by pbo on 23.07.19.
//

#include <LossLayer.hpp>
#include <Placeholder.hpp>
#include <CrossEntropyOp.hpp>
#include <MSEOp.hpp>


LossLayer::LossLayer(std::shared_ptr<AbstractLayer> input, std::shared_ptr<Graph> computeGraph, LossType losstype)
		: AbstractLayer(std::move(input), std::move(computeGraph)) {



	/*
	 * Initialization Labels
	 */
	Matrix tmp;


	/*
	 * Initialization of Loss Operation
	 */

	switch (losstype) {
		case LossType::CrossEntropy:
			_lossNode = OperationsFactory::createCrossEntropyOp(getComputeGraph(), getInputNode(), tmp);
			break;
		case LossType::MSE:
			_lossNode = OperationsFactory::createMSEOp(getComputeGraph(), getInputNode(), tmp);
			break;
		default:
			throw std::runtime_error(std::string("the selected LossType has yet not been Implemented in LossLayer class"));
	}

	setOutputNode(_lossNode);

}
float LossLayer::getLoss() {
	return _lossNode->getLoss();
}

void LossLayer::updateLabels(Matrix &newLabels) {
	if (std::dynamic_pointer_cast<LossFunction>(getOutputNode()) != nullptr) {
		std::dynamic_pointer_cast<LossFunction>(getOutputNode())->getLabels()->setForward(newLabels);
	}
}

const Matrix LossLayer::getPrediction() const {
    return _lossNode->getPrediction();
}

