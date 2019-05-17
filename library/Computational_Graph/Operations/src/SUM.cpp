//
// Created by phili on 10.05.2019.
//

#include "SUM.hpp"

void SUM::forwards(){
	float tmp =0;
	for(int i=0;i<getInputNodes().size();i++){
		_forwardCache(i)=getInputNodes().at(i)->getForwardData();
		tmp+= getInputNodes().at(i)->getForwardData();
	}
	setForwardData(tmp);
}
void SUM::backwards(float previousGradient) {
	for(int i =0;i<_forwardCache.size();i++){
		_gradients(i)=previousGradient;
	}

}
