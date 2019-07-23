//
// Created by phili on 11.05.2019.
//
#pragma once

#include <Node.hpp>
#include <chrono>
class Session {
public:
    Session(const std::shared_ptr<Node> &endNode);
    ~Session() = default;
	void run(std::vector<float> feed = {});

    int getForwardTime() const;

    int getBackwardsTime() const;

private:
	std::vector<std::shared_ptr<Node>> postOrderTraversal(const std::shared_ptr<Node> &endNode);

	std::vector<std::shared_ptr<Node>> preOrderTraversal(const std::shared_ptr<Node> &endNode);

	void backProp(std::shared_ptr<Node> &endNode);

	std::vector<std::shared_ptr<Node>> _postOrderTraversedList;
	std::vector<std::shared_ptr<Node>> _preOrderTraversedList;
	std::shared_ptr<Node> _endNode;

    int _forwardTime, _backwardsTime;
    std::chrono::time_point<std::chrono::system_clock> _start,_end;

};


