//
// Created by phili on 30.06.2019.
//

#pragma once

#include <Operation.hpp>

class Flatten : public Operation {
public:
	Flatten(std::shared_ptr<Node> X)
			: Operation(X) {	setOutputChannels(0);
	};

	~Flatten() = default;

	void forwards() override;

	void backwards() override;
	std::string printForward() override;

};