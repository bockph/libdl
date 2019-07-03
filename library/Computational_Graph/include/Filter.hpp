//
// Created by phili on 14.06.2019.
//
#pragma once

#include <Node.hpp>




class Filter : public Node {

public:
	Filter(Eigen::MatrixXf t,int dim=1,int channel=1);

	~Filter() = default;

	void backwards() override;

private:
    Eigen::MatrixXf _s1;
    Eigen::MatrixXf _v1;

};


