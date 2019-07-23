//
// Created by phili on 10.05.2019.
//
#pragma once


#include <Operation.hpp>

class MultiplicationOp : public Operation {
public:
	MultiplicationOp(std::shared_ptr<Node> X, std::shared_ptr<Node> W)
			: Operation(X, W) {
//	    if(X->getForward().cols()!=W->getForward().rows()){
//	        int cols =X->getForward().cols();
//            int rows =W->getForward().rows();
//            throw std::runtime_error("Multiplication Operation: X.cols() should equal W.rows()");
//	    }
	};

	~MultiplicationOp() = default;

	void forwards() override;

	void backwards() override;

};
