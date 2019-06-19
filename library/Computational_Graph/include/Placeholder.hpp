//
// Created by phili on 10.05.2019.
//

#pragma once

#include <Node.hpp>

class Placeholder : public Node {

public:
	Placeholder(Eigen::MatrixXf,int dim =1,int channel =1);

	void backwards() override;
};



