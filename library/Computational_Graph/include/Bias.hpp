//
// Created by phili on 21.05.2019.
//



#pragma once

#include <Node.hpp>
#include <Operation.hpp>
class Bias : public Node {

public:
	Bias(float t);
	Bias(Eigen::MatrixXf t);

	using Node::forwards;
	using Node::getInputNodes;
	void backwards(float previousGradient) override;
	void backwards() override;

};


