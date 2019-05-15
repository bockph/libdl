//
// Created by phili on 08.05.2019.
//
#pragma once
#include "Node.hpp"
#include <vector>
#include <memory>

class Operation: public Node
		{

public:
	Operation(std::vector<std::shared_ptr<Node>> inputNodes);

//	virtual void compute(){};
	using Node::compute;
//	using Node::addOutputNode;

	const std::vector<std::shared_ptr<Node>> &getInputNodes() override ;

private:
	std::vector<std::shared_ptr<Node>> _inputNodes;



};


