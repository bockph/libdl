//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "Placeholder.hpp"

Placeholder::Placeholder(float t){
	setForwardData(t);

//	graph->addPlaceholder(std::make_shared<Node>(this));
}
void Placeholder::backwards(float previousGradient) {

}
Placeholder::Placeholder(Eigen::MatrixXf m)
{
	setForward(m);
//	currentGradients(m.cols(),m.rows());

}
void Placeholder::backwards() {
/*
	std::cout<<"BackProp Input:"<<getCurrentGradients()<<std::endl;
*/

}