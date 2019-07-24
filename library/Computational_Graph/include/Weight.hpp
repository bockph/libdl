//
// Created by phili on 14.06.2019.
//
#pragma once

#include <Node.hpp>




class Weight : public Node {

public:
	Weight(Eigen::MatrixXf& t,int dim=1,int channel=1);

	~Weight() = default;

	void backwards() override;

    float getLearningRate() const;

    void setLearningRate(float learningRate);

private:
    Eigen::MatrixXf _s1;
    Eigen::MatrixXf _v1;
    float _learningRate;

};


