//
// Created by phili on 17.05.2019.
//

#pragma once

#include <Node.hpp>
#include <Operation.hpp>
class Weight : public Node {

public:
	Weight(float t);
	Weight(Eigen::MatrixXf t);

	using Node::forwards;
	using Node::getInputNodes;
	void backwards(float previousGradient) override;
	void backwards(bool first = false) override;

};


