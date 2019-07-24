//
// Created by phili on 11.05.2019.
//

#include "Session.hpp"
#include <Variable.hpp>
#include <iostream>
#include <IO.hpp>


Session::Session(const std::shared_ptr<Node> &endNode, hyperParameters params)
        :
        _postOrderTraversedList(postOrderTraversal(endNode))
        , _endNode(endNode)
        , _params(params){

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

/*std::vector<std::shared_ptr<Node>> Session::preOrderTraversal(const std::shared_ptr<Node> &endNode) {
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
}*/

void Session::backProp(std::shared_ptr<Node> &endNode) {

	endNode->backwards();

	auto tmp = endNode->getInputNodes();
	for (int i = 0; i < tmp.size(); i++) {
		backProp(tmp.at(i));

	}

}

void Session::run() {


    _start = std::chrono::system_clock::now();

	for (std::shared_ptr<Node> node: _postOrderTraversedList) {
        if(std::dynamic_pointer_cast<Variable>(node)!= nullptr){
            std::dynamic_pointer_cast<Variable>(node)->setHyperParameters(_params);
        }

		node->forwards();
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

bool Session::writeVariables(std::string dir) {

    int idx =0;
    for (std::shared_ptr<Node> node: _postOrderTraversedList) {
        if(std::dynamic_pointer_cast<Variable>(node)!= nullptr)
            if(!write_binary(dir+std::to_string(idx)+std::string(".bin"), std::dynamic_pointer_cast<Variable>(node)->getForward()))
                return false;
        idx++;
    }
    return true;
}

bool Session::readVariables(std::string dir) {
    int idx =0;
    for (std::shared_ptr<Node> node: _postOrderTraversedList) {
        if(std::dynamic_pointer_cast<Variable>(node)!= nullptr){

            Matrix tmpStore;
            if(!read_binary(dir+std::to_string(idx)+std::string(".bin"), tmpStore))
                return false;
            std::dynamic_pointer_cast<Variable>(node)->setForward(tmpStore);

        }

        idx++;
    }
    return true;}

int Session::getForwardTime() const {
    return _forwardTime;
}

int Session::getBackwardsTime() const {
    return _backwardsTime;
}

const hyperParameters &Session::getParams() const {
    return _params;
}

void Session::setParams(const hyperParameters &params) {
    _params = params;
}

