//
// Created by phili on 21.05.2019.
//



#pragma once

#include <Node.hpp>

class Bias : public Node {

public:
	Bias(Eigen::MatrixXf t, int channels =0);

	void backwards() override;
private:
    Eigen::MatrixXf _s1;
    Eigen::MatrixXf _v1;
};


