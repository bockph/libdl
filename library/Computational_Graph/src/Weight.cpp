//
// Created by phili on 17.05.2019.
//

#include <iostream>
#include "Weight.hpp"

Weight::Weight(float m){
	setForwardData(m);
//	graph->addWeight(std::make_shared<Node>(this));
}
Weight::Weight(Eigen::MatrixXf m){
	setForward(m);
//	currentGradients(m.rows(),m.cols());
}

void Weight::backwards(float previousGradient) {
	setForwardData(getForwardData()-0.1*previousGradient);
}
void Weight::backwards() {
/*
	std::cout<<"BackProp Weight:"<<getCurrentGradients()<<std::endl;
*/
	setForward(getForward()-50*getCurrentGradients());
}