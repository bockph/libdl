//
// Created by phili on 27.07.2019.
//

#include "LossFunction.hpp"

const std::shared_ptr<Placeholder> &LossFunction::getLabels() const {
	return _labels;
}
