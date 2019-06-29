//
// Created by phili on 10.05.2019.
//

#include "Placeholder.hpp"


Placeholder::Placeholder(Eigen::MatrixXf m,int dim,int channel) {
	setForward(m);
	setOutputDim(dim);
    setOutputChannels(channel);
}

void Placeholder::backwards() {}