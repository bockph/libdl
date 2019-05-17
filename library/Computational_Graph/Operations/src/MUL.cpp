//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "MUL.hpp"

void MUL::forwards(){

	float tmp =1;

	for(int i=0;i<getInputNodes().size();i++){
		_forwardCache(i)=getInputNodes().at(i)->getForwardData();
		tmp*= getInputNodes().at(i)->getForwardData();
	}
	setForwardData(tmp);
};
void MUL::backwards(float previousGradient) {
	for(int i =0;i<_forwardCache.size();i++){
		_gradients(i)=previousGradient;
		for(int s =0;s<_forwardCache.size();s++){
			if(s!=i){
				_gradients(i)*=_forwardCache(s);
			}
		}
	}
}