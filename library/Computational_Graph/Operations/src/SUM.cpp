//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "SUM.hpp"

void SUM::forwards(){
	/*float tmp =0;
	for(int i=0;i<getInputNodes().size();i++){
		_forwardCache(i)=getInputNodes().at(i)->getForwardData();
		tmp+= getInputNodes().at(i)->getForwardData();
	}
	setForwardData(tmp);*/

	int rowsA =getInputA()->getForward().rows();
	int rowsB=getInputB()->getForward().rows();

if(rowsA!=rowsB)
	setForward(getInputA()->getForward()+getInputB()->getForward().replicate(rowsA,1));
else
	setForward(getInputA()->getForward()+getInputB()->getForward());


}
void SUM::backwards(float previousGradient) {
	for(int i =0;i<_forwardCache.size();i++){
		_gradients(i)=previousGradient;
	}

}
void SUM::backwards(){
	getInputB()->setCurrentGradients(getCurrentGradients());
	getInputA()->setCurrentGradients(getCurrentGradients());

}