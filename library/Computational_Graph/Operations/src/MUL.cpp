//
// Created by phili on 10.05.2019.
//

#include "MUL.hpp"

void MUL::compute(){
	float tmp =1;
	for(std::shared_ptr<Node> input: getInputNodes()){
		tmp*= input->getDatavalue();
	}
	setDatavalue(tmp);
};
