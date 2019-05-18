//
// Created by phili on 17.05.2019.
//

#pragma once


#include <Operation.hpp>

class MSE : public Operation {
public:
	MSE(std::vector<std::shared_ptr<Node>> inputNodes):Operation(inputNodes){};//;
	~MSE()=default;
	void forwards() override;
	void backwards(float previousGradient) override;


};
