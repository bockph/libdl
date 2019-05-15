//
// Created by phili on 10.05.2019.
//

#include "Graph.hpp"

//const std::vector<std::shared_ptr<Node>> &Graph::getPlaceholder() const {
//	return _placeholder;
//}
//
//void Graph::addPlaceholder(std::shared_ptr<Node> placeholder) {
//	_placeholder.push_back(placeholder);
//}
//
//const std::vector<std::shared_ptr<Node>> &Graph::getOperations() const {
//	return _operations;
//}
//
//void Graph::addOperation( std::shared_ptr<Node> operation) {
//	_operations.push_back(operation);
//}

const std::vector<std::shared_ptr<Placeholder>> &Graph::getPlaceholder() const {
	return _placeholder;
}

void Graph::addPlaceholder(std::shared_ptr<Placeholder> placeholder) {
	_placeholder.push_back(placeholder);
}

const std::vector<std::shared_ptr<Operation>> &Graph::getOperations() const {
	return _operations;
}

void Graph::addOperation( std::shared_ptr<Operation> operation) {
	_operations.push_back(operation);
}
