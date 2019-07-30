//
// Created by phili on 08.05.2019.
//
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <commonDatatypes.hpp>



/*!
 * This class represents the Base Class of all Nodes of the Computational Graph.
 */
class Node {
public:
	virtual ~Node()=default;
private:
	/*!
	 * Contains the Output of this Node which equals to the Data Input of the following Node
	 * Each sample that was computed is a row-Vector and therefore represented by one row of the Matrix
	 */
	Matrix _forward;
	/*!
	 * contains the Gradients of the previous Node, if the Node is the EndNode, it should contain a Matrix of Ones with
	 * the Dimensions of _forward. Further, each row represents the Gradient for one sample.
	 */
	Matrix _previousGradients;

	/*!
	 * contains the amount of Channels the output of this Node has
	 */
    int _outputChannels;


    /*
     * Getters & Setters
     */
public:

	const Matrix &getForward() const;

	void setForward(const Matrix &forward);

	const Matrix &getPreviousGradients() const;

	void setPreviousGradients(const Matrix &previousGradients);

	void setOutputChannels(int outputChannels);

	int getOutputChannels() const;



};


