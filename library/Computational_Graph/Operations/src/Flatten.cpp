//
// Created by phili on 30.06.2019.
//

#include "Flatten.hpp"



#include <iostream>


void Flatten::forwards() {
    //General Stuff for Operations
    beforeForward();
	setForward(getInputA()->getForward());

};

void Flatten::backwards() {
	getInputA()->setCurrentGradients(getCurrentGradients());
}

