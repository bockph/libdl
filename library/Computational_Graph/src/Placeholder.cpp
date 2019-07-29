//
// Created by phili on 10.05.2019.
//

#include "Placeholder.hpp"


Placeholder::Placeholder(Matrix& batch,int channel) {
	setForward(batch);
	setOutputChannels(channel);
}
Placeholder::Placeholder(Matrix& labels) {
	setForward(labels);
}
