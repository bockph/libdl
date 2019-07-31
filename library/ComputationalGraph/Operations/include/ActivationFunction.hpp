//
// Created by phili on 27.07.2019.
//

#pragma once

#include <Operation.hpp>

/*!
 * An ActivationFunction operation object introduces non linearity into the computational graph
 */
class ActivationFunction : public Operation {
public:
    /*!
     * - creates the operation Object from the input Node
     * - the channels are simply forwarded from the input Node
     * @param X input Node
     */
	ActivationFunction(std::shared_ptr<Node> X);

	~ActivationFunction() = default;//! default destructor


};


