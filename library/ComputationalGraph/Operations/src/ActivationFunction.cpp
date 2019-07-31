//
// Created by phili on 27.07.2019.
//

#include "ActivationFunction.hpp"

ActivationFunction::ActivationFunction(std::shared_ptr<Node> X)
        : Operation(X, X->getOutputChannels()) {}