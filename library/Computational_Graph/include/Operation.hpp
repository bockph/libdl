//
// Created by phili on 08.05.2019.
//
#pragma once

#include "Node.hpp"
#include <vector>
#include <memory>
#include <chrono>
#include <ctime>
/*!
 * This class defines the interface each Operation of the Computational Graph has to implement
 */
class Operation : public Node {

public:
	/*!
	 * stores the obligatory Input Node for a operation and sets the _outputChannel for this Operation. Further it
	 * stores the _outputChannel of the input Node as _inputChannel
	 * @param input the inputNode which might be an Placeholder or an Operation
	 * @param outputChannel the #outputChannel of this Operation
	 */
	Operation(std::shared_ptr<Node> input,int outputChannel);
	/*!
	 * Default destructor
	 */
    virtual ~Operation() = default;

	/*!
	 * calculates the outputValue of this Operation using the Input Nodes.
	 *
	 * Each implementation must set the  member variable _forward by calling setForward(Matrix calculatedOutputValue)
	 */
	virtual void forwardPass() {};

	/*!
	 * calculates the gradients for the operation inputs w.r.t. the last Node in the Computational Grpah.
	 * In order to do so it uses the member variable _previousGradients of Class Node, which contains the Gradients of
	 * the previous operation.
	 *
	 * Each implementation must set the member variable _previousGradients of the operations Inputs by calling
	 * [AllInputNodes]->setPreviousGradients(Matrix calculatedGradients)
	 */
	virtual void backwardPass() {};

	/*!
	 * executes the forwardPass and stores the computation time
	 * @return the computation Time
	 */
	int forwardPassWithMeasurement();

	/*!
	 * executes the backwardPass and stores the computation time
	 * @return the computation Time
	 */
	int backwardPassWithMeasurement();


	/*!
	 * this does only work if forwardPassWithMeasurement() is called
	 * @return computation Time of forwardPass
	 */
    int getForwardTime() const;

	/*!
	 * this does only work if backwardPassWithMeasurement() is called
	 * @return computation Time of backwardPass
	 */
    int getBackwardsTime() const;

private:
	std::shared_ptr<Node> _input;//! This can be an operation or a Placeholder;
	int _inputChannels; //! stores the _outputChannel of _input



    int _forwardTime, _backwardsTime; //!stores calculated Measuremnts
    std::chrono::time_point<std::chrono::system_clock> _start,_end; //! set time_points for Measurements
/*!
	 * starts Time measurement
	 */
	void startTimeMeasurement();
	/*!
	 * stops Time measurement
	 * @param function if function =0, measurement is stored in _forwardTime, if function =1 measurement is stored in
	 * _backwardsTime
	 * @return the measurement
	 */
	int stopTimeMeasurement(char function);


    /*
     * Getters & Setters
     */
public:
	const std::shared_ptr<Node> &getInput() const;
	int getInputChannels() const;
};


