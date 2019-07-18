//
// Created by phili on 30.06.2019.
//

#include "Flatten.hpp"
//
// Created by pbo on 18.06.19.
//

#include "ReLu.hpp"


#include <iostream>


void Flatten::forwards() {
	/*
 * GENERALL STUFF
 */
//    setOutputChannels(getInputA()->getOutputChannels());
/*
	*
 */
	setForward(getInputA()->getForward());

};

void Flatten::backwards() {

	getInputA()->setCurrentGradients(getCurrentGradients());
}

