//
// Created by pbo on 17.06.19.
//



#pragma once


#include "NormalFunction.hpp"

class MaxPoolOp : public NormalFunction {
public:
	MaxPoolOp(std::shared_ptr<Node> X, int windowSize, int stride = 1);

	~MaxPoolOp() override = default;

	void forwardPass() override;

	void backwardPass() override;

	const Eigen::MatrixXf &getMaxIndexMatrix() const;

	void setMaxIndexMatrix(const Eigen::MatrixXf &maxIndexMatrix);

private:
	int _stride;
	int _windowSize;
	Eigen::MatrixXf _maxIndexMatrix;
};
