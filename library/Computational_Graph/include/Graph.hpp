//
// Created by phili on 10.05.2019.
//

#pragma once

#include <vector>
#include <Node.hpp>
#include <memory>
#include <Operation.hpp>
#include <Placeholder.hpp>

class Graph {

public:
	const std::vector<std::shared_ptr<Placeholder>> &getPlaceholder() const;
//
	void addPlaceholder(std::shared_ptr<Placeholder> placeholder);
//
	const std::vector<std::shared_ptr<Operation>> &getOperations() const;

	void addOperation(std::shared_ptr<Operation> operation);

//private:
	std::vector<std::shared_ptr<Placeholder>> _placeholder;
	std::vector<std::shared_ptr<Operation>> _operations;

};



