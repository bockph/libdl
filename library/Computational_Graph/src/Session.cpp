//
// Created by phili on 11.05.2019.
//

#include "Session.hpp"

Session::Session(const std::shared_ptr<Node> &endNode, std::unique_ptr<Graph> graph)
		:
		_graph(std::move(graph))
		, _postOrderTraversedList(postOrderTraversal(endNode))
		, _preOrderTraversedList(preOrderTraversal(endNode))
		, _endNode(endNode){

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

void Session::backProp(std::shared_ptr<Node> &endNode, float gradient,bool first) {
//	endNode->backwards(gradient);
if(first)
	endNode->backwards(true);
else
	endNode->backwards();

	auto tmp = endNode->getInputNodes();
//	if (!tmp.empty()) {
		for (int i = 0; i < tmp.size(); i++) {
//			backProp(tmp.at(i), endNode->_gradients(i));
			backProp(tmp.at(i));//, endNode->_gradients(i));

		}
//	}
}

void Session::run(std::vector<float> feed) {

	//TODO: write method to fill placeholders with the feed,
	//maybe some vector of pairs with <Node,float>??

	for (std::shared_ptr<Node> operation: _postOrderTraversedList) {
		operation->forwards();
	}
	Eigen::MatrixXf tmp = _endNode->getForward();
	tmp.setOnes();
	_endNode->setCurrentGradients(tmp);
	backProp(_endNode,0,true);//,1);

//	for(int i=0;i<_preOrderTraversedList.size();i++){
//		if(i==0)_preOrderTraversedList.at(i)->backwards(1);
//		else{
//			_preOrderTraversedList.at(i)->backwards(_preOrderTraversedList.at(i-1)->getBackwardData());
//		}
//
//	}


}

