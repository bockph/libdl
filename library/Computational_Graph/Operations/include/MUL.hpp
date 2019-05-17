//
// Created by phili on 10.05.2019.
//
#pragma once


#include <Operation.hpp>

class MUL : public Operation {
public:
	MUL(std::vector<std::shared_ptr<Node>> inputNodes):Operation(inputNodes){};//;
	~MUL()=default;
	void forwards() override;
	void backwards(float previousGradient) override;


};
