//
// Created by phili on 08.05.2019.
//

#include <iostream>
#include "Operation.hpp"

 Operation::Operation(std::vector<std::shared_ptr<Node>> inputNodes):
 //store all references of input Nodes
_inputNodes(inputNodes)


{
	//add this Node as a Output Node for all reference inputNodes
	for(std::shared_ptr<Node> input: _inputNodes){
		auto tmp(std::make_shared<Operation>(*this));
		input->addOutputNode( tmp);//std::make_shared<Operation>(this));
	}

	_forwardCache=Eigen::VectorXf(inputNodes.size());

	_gradients=Eigen::VectorXf(inputNodes.size());

	//append to default active graph
//	graph->addOperation(std::make_shared<Node>(this));//setOperations(this);
}
Operation::Operation(std::shared_ptr<Node> X, std::shared_ptr<Node> W)
{
	setInputA(X);
	setInputB(W);

	_inputNodes.push_back(X);
	_inputNodes.push_back(W);
	//add this Node as a Output Node for all reference inputNodes
	auto tmp(std::make_shared<Operation>(*this));
	X->addOutputNode( tmp);
	W->addOutputNode( tmp);


	//just taken from above
	_forwardCache=Eigen::VectorXf(_inputNodes.size());

	_gradients=Eigen::VectorXf(_inputNodes.size());
}
Operation::Operation(std::shared_ptr<Node> X)
{
	_inputNodes.push_back(X);
	setInputA(X);
	//add this Node as a Output Node for all reference inputNodes
	auto tmp(std::make_shared<Operation>(*this));
	X->addOutputNode(tmp);



	//just taken from above
	_forwardCache=Eigen::VectorXf(_inputNodes.size());

	_gradients=Eigen::VectorXf(_inputNodes.size());
}
const std::vector<std::shared_ptr<Node>> &Operation::getInputNodes()  {
	return _inputNodes;
}




