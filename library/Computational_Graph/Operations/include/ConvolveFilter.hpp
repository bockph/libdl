//
// Created by phili on 14.06.2019.
//

#pragma once


#include <Operation.hpp>

class ConvolveFilter : public Operation {
public:
	ConvolveFilter(std::shared_ptr<Node> X, std::shared_ptr<Node> W,int stride =1)
			: Operation(X, W),_stride(stride) {};

	~ConvolveFilter() = default;

	void forwards() override;

	void backwards() override;
	std::string printForward() override;
private:
	int _stride;
};
