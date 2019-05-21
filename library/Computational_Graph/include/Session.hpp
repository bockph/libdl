//
// Created by phili on 11.05.2019.
//
#include <Graph.hpp>

#pragma once

class Session {
public:
	Session(const std::shared_ptr<Node> &endNode, std::unique_ptr<Graph> graph);

	void run(std::vector<float> feed = {});

private:
	std::vector<std::shared_ptr<Node>> postOrderTraversal(const std::shared_ptr<Node> &endNode);

	std::vector<std::shared_ptr<Node>> preOrderTraversal(const std::shared_ptr<Node> &endNode);

	void backProp(std::shared_ptr<Node> &endNode);

	std::vector<std::shared_ptr<Node>> _postOrderTraversedList;
	std::vector<std::shared_ptr<Node>> _preOrderTraversedList;
	std::unique_ptr<Graph> _graph;
	std::shared_ptr<Node> _endNode;

};


