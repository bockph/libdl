//
// Created by phili on 10.05.2019.
//
#pragma once


#include <Operation.hpp>

class MUL : public Operation {
public:
	MUL(std::vector<std::shared_ptr<Node>> inputNodes):Operation(inputNodes){};//;
	MUL(std::shared_ptr<Node> X, std::shared_ptr<Node> W):Operation(X,W){};
	~MUL()=default;
	void forwards() override;
	void backwards(float previousGradient) override;
	void backwards(bool first = false) override;


};
