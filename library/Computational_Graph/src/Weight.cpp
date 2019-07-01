//
// Created by phili on 17.05.2019.
//

#include <iostream>
#include "Weight.hpp"


Weight::Weight(Eigen::MatrixXf m) {
	setForward(m);
}

void Weight::backwards() {

	setForward(getForward() - 0.01 * getCurrentGradients());
}