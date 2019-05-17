//
// Created by phili on 10.05.2019.
//

#pragma once

#include <Node.hpp>
#include <Operation.hpp>
class Placeholder : public Node {

public:
	Placeholder(float t);
//	using Node::addOutputNode;
	using Node::forwards;
//	using Node::getDatavalue;
//	using Node::setDatavalue;
	using Node::getInputNodes;
//	using Node::getType;
//protected:
//	 const int _id=2;

};



