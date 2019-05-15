//
// Created by phili on 11.05.2019.
//

#include "Session.hpp"
Session::Session(const std::shared_ptr<Node> &endNode,std::unique_ptr<Graph> graph):
_graph(std::move(graph))
,_postOrderTraversedList(postOrderTraversal(endNode))

{

}

std::vector<std::shared_ptr<Node>> Session::postOrderTraversal(const std::shared_ptr<Node> &endNode){
	std::vector<std::shared_ptr<Node>> toReturn;



		if(!endNode->getInputNodes().empty()){
			std::vector<std::shared_ptr<Node>> tmp;
			for(std::shared_ptr<Node> input: endNode->getInputNodes()) {
//				if(Operation* o = dynamic_cast<Operation*>(input.get())){
//					Operation test = static_cast<Operation&>(*input);

//				tmp =postOrderTraversal(std::make_shared<Operation>(*o));
				tmp = postOrderTraversal(input);
				toReturn.insert(std::end(toReturn), std::begin(tmp), std::end(tmp));
//					toReturn.push_back(std::make_shared<Operation>(*o));
//				}


			}
		}
			toReturn.push_back(endNode);



	return toReturn;
}
void Session::run(std::vector<float> feed) {

	//TODO: write method to fill placeholders with the feed,
	//maybe some vector of pairs with <Node,float>??

	for(std::shared_ptr<Node> operation: _postOrderTraversedList){
		operation->compute();
//		operation->setDatavalue(2);
	}

}

