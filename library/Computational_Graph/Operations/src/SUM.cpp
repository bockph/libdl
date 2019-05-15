//
// Created by phili on 10.05.2019.
//

#include "SUM.hpp"

void SUM::compute(){
	float tmp =0;
	for(std::shared_ptr<Node> input: getInputNodes()){
		tmp+= input->getDatavalue();
	}
	setDatavalue(tmp);
};
