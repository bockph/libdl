//
// Created by pbo on 17.06.19.
//



#pragma once


#include "NormalFunction.hpp"

/*!
 *  Implements a MaxPooling operation
 */
class MaxPoolOp : public NormalFunction {
public:
    /*!
    * - creates a Normal Function and sets the outputChannel to the number of channels of the input Node
    *
    * @param X input Noode
    * @param windowDim dimensions of the square window that is sliding over the samples taking the max value
    * @param stride stride
    */
    MaxPoolOp(std::shared_ptr<Node> X, int windowDim, int stride = 1);

    ~MaxPoolOp() override = default;//!default destructor

    /*!
     * applies Maxpooling to input X usig the window dimension
     * stores additionaly an index matrix of dim(X) containing at the index of the Max Values a one
     */
    void forwardPass() override;

    /*!
     * calculates the gradient w.r.t the input isung the index Matrix
     */
    void backwardPass() override;

private:
    int _stride; //! striding value
    int _windowSize; //! dimensions of the square window that is sliding over the samples taking the max value
    Eigen::MatrixXf _maxIndexMatrix;//! stores the indices of the max Values

    /*
     * Getters & Setters
     */
public:

    const Eigen::MatrixXf &getMaxIndexMatrix() const;

    void setMaxIndexMatrix(const Eigen::MatrixXf &maxIndexMatrix);

};
