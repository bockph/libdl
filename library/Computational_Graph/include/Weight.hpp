//
// Created by phili on 17.05.2019.
//

#pragma once

#include <Node.hpp>

class Weight : public Node {

public:
	Weight(Eigen::MatrixXf& t);

	~Weight() = default;

	void backwards() override;

private:
    Eigen::MatrixXf _s1;
    Eigen::MatrixXf _v1;
};


