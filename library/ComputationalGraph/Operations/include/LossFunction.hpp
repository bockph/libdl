//
// Created by phili on 27.07.2019.
//

#pragma once

#include <Operation.hpp>
#include <Placeholder.hpp>

/*!
 * A LossFunction operation object represents operation to calculate the loss of a sample computation compared to its ground truth
 */
class LossFunction : public Operation {
public:
    /*!
     *  - creates the operation Object using the the obligatory Input Node.
     *  - stores the labels for this operation
     *  - sets the output Channel of the operation Object to a random Value, as at this point channels do not matter anymore
     *
     * @param X the input Node
     * @param labels the ground truth to the input samples
     */
    LossFunction(std::shared_ptr<Node> X, std::shared_ptr<Placeholder> labels);

    ~LossFunction() override = default; //! default Destructor

private:
    std::shared_ptr<Placeholder> _labels; //!the ground truth to the input samples

    /*
     * Getters & Setters
     */
public:
    const std::shared_ptr<Placeholder> &getLabels() const;

    const Matrix getPrediction() const; //!returns the forward pass of the input Node aka the prediction

    const float getLoss() const; //!returns the current loss

};


