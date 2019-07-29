//
// Created by phili on 10.05.2019.
//

#pragma once

#include <Node.hpp>

/*!
 * This Node represents a Placeholder for a fixed Input into the Computational Graph
 * In terms of a Neural Networks, this class holds the functionality for inputing Samples or Labels to corresponding
 * Samples
 */
class Placeholder : public Node {
public:
	/*!
	 * constructs a mini-batch a sets the output to batch. Further the number of output channels is set to the number
	 * of channels of each samples
	 * @param batch a whole mini-batch, each row contains one sample
	 * @param channel number of channels of each sample
	 */
	Placeholder(Matrix &batch, int channel);

	/*!
	 * constructs a Placeholder that contains the labels to a previously created mini-batch.
	 * @param labels each row corresponds to the label of one sample, while the Dimension of each row is the #labels,
	 * this way the corresponding label is identified by the collumn that holds a "1"
	 */
	Placeholder(Matrix &labels);

	/*!
	 * Default Destructor
	 */
	~Placeholder() = default;
};



