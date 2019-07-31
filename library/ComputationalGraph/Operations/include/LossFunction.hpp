//
// Created by phili on 27.07.2019.
//

#pragma once

#include <Operation.hpp>
#include <Placeholder.hpp>
class LossFunction: public Operation {
public:
	LossFunction(std::shared_ptr<Node> X, std::shared_ptr<Placeholder> labels);
	~LossFunction() override=default;

private:
	std::shared_ptr<Placeholder> _labels;

	/*
	 * Getters & Setters
	 */
public:
	const std::shared_ptr<Placeholder> &getLabels() const;
	const Matrix getPrediction() const;

};


