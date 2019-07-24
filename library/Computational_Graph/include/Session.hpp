//
// Created by phili on 11.05.2019.
//
#pragma once

#include <Node.hpp>
#include <chrono>
#include <commonDatatypes.hpp>
class Session {
public:
//TODO set as deprecated
    Session(const std::shared_ptr<Node> &endNode, hyperParameters params=hyperParameters());
    ~Session() = default;
	void run();

    int getForwardTime() const;

    int getBackwardsTime() const;

    bool writeVariables(std::string dir);
    bool readVariables(std::string dir);

private:
	std::vector<std::shared_ptr<Node>> postOrderTraversal(const std::shared_ptr<Node> &endNode);

//	std::vector<std::shared_ptr<Node>> preOrderTraversal(const std::shared_ptr<Node> &endNode);

	void backProp(std::shared_ptr<Node> &endNode);

	std::vector<std::shared_ptr<Node>> _postOrderTraversedList;
//	std::vector<std::shared_ptr<Node>> _preOrderTraversedList;
	std::shared_ptr<Node> _endNode;

    int _forwardTime, _backwardsTime;
    std::chrono::time_point<std::chrono::system_clock> _start,_end;
public:
    const hyperParameters &getParams() const;

    void setParams(const hyperParameters &params);

private:
    hyperParameters _params;


};


