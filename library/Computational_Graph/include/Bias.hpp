//
// Created by phili on 21.05.2019.
//



#pragma once

#include <Node.hpp>

class Bias : public Node {

public:
	Bias(Eigen::MatrixXf t);

	void backwards() override;

};


