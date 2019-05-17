//
// Created by phili on 16.05.2019.
//
#pragma once


#include <Operation.hpp>

class Sigmoid : public Operation {
public:
	Sigmoid(std::vector<std::shared_ptr<Node>> inputNodes):Operation(inputNodes){};//;
	~Sigmoid()=default;
	void forwards() override;
	void backwards(float previousGradient) override;
	float sigmoid(float a);


};
