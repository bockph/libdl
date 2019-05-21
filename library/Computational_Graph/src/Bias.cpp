//
// Created by phili on 21.05.2019.
//


#include <iostream>
#include "Bias.hpp"

Bias::Bias(float m){
	setForwardData(m);
//	graph->addWeight(std::make_shared<Node>(this));
}
Bias::Bias(Eigen::MatrixXf m){
	setForward(m);
//	currentGradients(m.rows(),m.cols());
}

void Bias::backwards(float previousGradient) {
	setForwardData(getForwardData()-0.1*previousGradient);
}
void Bias::backwards() {
/*
	std::cout<<"BackProp Bias:"<<getCurrentGradients()<<std::endl;
*/
	int rowsG =getCurrentGradients().rows();
	int rowsCurrent=getForward().rows();
	if(rowsCurrent!=rowsG)
	setForward(getForward().replicate(rowsG,1)-50*getCurrentGradients());
	else
		setForward(getForward()-50*getCurrentGradients());

}