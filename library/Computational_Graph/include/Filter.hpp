//
// Created by phili on 14.06.2019.
//
#pragma once

#include <Node.hpp>




class Filter : public Node {

public:
	Filter(Eigen::MatrixXf t);

	~Filter() = default;

	void backwards() override;


};


