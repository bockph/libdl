//
// Created by phili on 17.05.2019.
//

#include "Weight.hpp"

Weight::Weight(float t){
	setForwardData(t);
//	graph->addWeight(std::make_shared<Node>(this));
}


void Weight::backwards(float previousGradient) {
	setForwardData(getForwardData()-0.1*previousGradient);
}