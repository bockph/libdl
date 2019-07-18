//
// Created by phili on 11.05.2019.
//

#include "Session.hpp"

Session::Session(const std::shared_ptr<Node> &endNode, std::unique_ptr<Graph> graph)
		:
		_graph(std::move(graph))
		, _postOrderTraversedList(postOrderTraversal(endNode))
		, _preOrderTraversedList(preOrderTraversal(endNode))
		, _endNode(endNode) {

}

std::vector<std::shared_ptr<Node>> Session::postOrderTraversal(const std::shared_ptr<Node> &endNode) {
	std::vector<std::shared_ptr<Node>> toReturn;
	if (!endNode->getInputNodes().empty()) {
		std::vector<std::shared_ptr<Node>> tmp;
		for (std::shared_ptr<Node> input: endNode->getInputNodes()) {
			tmp = postOrderTraversal(input);
			toReturn.insert(std::end(toReturn), std::begin(tmp), std::end(tmp));
		}
	}
	toReturn.push_back(endNode);
	return toReturn;
}

std::vector<std::shared_ptr<Node>> Session::preOrderTraversal(const std::shared_ptr<Node> &endNode) {
	std::vector<std::shared_ptr<Node>> toReturn;
	toReturn.push_back(endNode);
	if (!endNode->getInputNodes().empty()) {
		std::vector<std::shared_ptr<Node>> tmp;
		for (std::shared_ptr<Node> input: endNode->getInputNodes()) {
			tmp = preOrderTraversal(input);
			toReturn.insert(std::end(toReturn), std::begin(tmp), std::end(tmp));
		}
	}
	return toReturn;
}

void Session::backProp(std::shared_ptr<Node> &endNode) {

	endNode->backwards();

	auto tmp = endNode->getInputNodes();
	for (int i = 0; i < tmp.size(); i++) {
		backProp(tmp.at(i));

	}

}

void Session::run(std::vector<float> feed) {

	//TODO: write method to fill placeholders with the feed,
	//maybe some vector of pairs with <Node,float>??
    _start = std::chrono::system_clock::now();

	for (std::shared_ptr<Node> operation: _postOrderTraversedList) {
		operation->forwards();
	}
    _end = std::chrono::system_clock::now();

    int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
            (_end-_start).count();

   _forwardTime=elapsed_seconds;



    _start = std::chrono::system_clock::now();

    Eigen::MatrixXf tmp = _endNode->getForward();
	tmp.setOnes();
	_endNode->setCurrentGradients(tmp);
	backProp(_endNode);
    _end = std::chrono::system_clock::now();

     elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
            (_end-_start).count();

    _backwardsTime=elapsed_seconds;


}

int Session::getForwardTime() const {
    return _forwardTime;
}

int Session::getBackwardsTime() const {
    return _backwardsTime;
}

