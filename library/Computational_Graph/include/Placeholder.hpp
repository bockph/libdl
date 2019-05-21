//
// Created by phili on 10.05.2019.
//

#pragma once

#include <Node.hpp>
#include <Operation.hpp>
class Placeholder : public Node {

public:
	Placeholder(float t);
	Placeholder(Eigen::MatrixXf);
	using Node::forwards;
	using Node::getInputNodes;
	using Node::_forward;
	void backwards(float previousGradient) override;
	void backwards() override;

};



