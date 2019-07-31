//
// Created by phili on 27.07.2019.
//

#pragma once

#include <Operation.hpp>
#include <Parameter.hpp>

class NormalFunction : public Operation {
public:
	NormalFunction(std::shared_ptr<Node> X, std::shared_ptr<Parameter> parameter, int outputChannels);

	NormalFunction(std::shared_ptr<Node> X, int outputChannels);

	~NormalFunction() override = default;

private:
	std::shared_ptr<Parameter> _parameter;

	/*
	 * Getters & Setters
	 */
public:
	const std::shared_ptr<Parameter> &getParameter() const;
};


