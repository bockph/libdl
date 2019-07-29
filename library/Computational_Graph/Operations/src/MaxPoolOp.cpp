//
// Created by pbo on 17.06.19.
//

#include <iostream>
#include "MaxPoolOp.hpp"

void MaxPoolOp::forwardPass() {

	int batchSize = getInput()->getForward().rows();
	int imgSize = getInput()->getForward().cols() / getInputChannels();
	int imgDim = std::sqrt(imgSize);
	int outputDim = std::floor((imgDim - _windowSize) / _stride) + 1;

	Eigen::MatrixXf::Index maxRow, maxCol;


	Eigen::MatrixXf outputMatrix = Eigen::MatrixXf::Zero(batchSize, outputDim * outputDim * getOutputChannels());
	Eigen::MatrixXf maxIndexMatrix = Eigen::MatrixXf::Zero(batchSize, getInput()->getForward().cols());


	//loop over all samples :
	for (int i = 0; i < batchSize; i++) {
		//loop over all channels
		for (int c = 0; c < getInputChannels(); c++) {
			//get the Current Channel of the Current Sample
			Eigen::MatrixXf currentChannelOfSample = getInput()->getForward().block(i, imgSize * c, 1, imgSize)
					.reshaped(imgDim, imgDim);

			//get the current block of the maxIndexMatrix
			Eigen::MatrixXf currentMaxIndexMatrix = maxIndexMatrix.block(i, imgSize * c, 1, imgSize)
					.reshaped(imgDim, imgDim);


			for (int x = 0; x < outputDim; x++) {
				for (int y = 0; y < outputDim; y++) {
					int x_stride = x * _stride;
					int y_stride = y * _stride;
					//apply maxPooling
					outputMatrix(i, y + x * outputDim + c * outputDim * outputDim) =
							currentChannelOfSample.transpose().block(x_stride, y_stride, 2, 2).maxCoeff(&maxRow, &maxCol);
					//store the index of the maximum Value
					currentMaxIndexMatrix.transpose().block(x_stride, y_stride, 2, 2)
							(maxRow, maxCol) = 1;

				}
			}

			maxIndexMatrix.block(i,
					imgSize * c, 1, imgSize) = currentMaxIndexMatrix.reshaped(1, imgSize).block(0, 0, 1, imgSize);

		}


	}
	setMaxIndexMatrix(maxIndexMatrix);
	setForward(outputMatrix);

};

void MaxPoolOp::backwardPass() {

	int batchSize = getInput()->getForward().rows();
	int imgSize = getInput()->getForward().cols() / getInputChannels();
	int imgDim = std::sqrt(imgSize);
	int outputDim = std::floor((imgDim - _windowSize) / _stride) + 1;


	Eigen::MatrixXf dX = Eigen::MatrixXf::Zero(batchSize, imgSize * getInputChannels());


	//loop over all samples :
	for (int i = 0; i < batchSize; i++) {
		//loop over all channels
		for (int c = 0; c < getInputChannels(); c++) {
			//get the maxIndexMatrix for current sample & channel
			Eigen::MatrixXf currentMaxIndexMatrix = getMaxIndexMatrix()
					.block(i, imgSize * c, 1, imgSize).reshaped(imgDim, imgDim);
			//apply Gradients to corrresponding maxIndex
			for (int x = 0; x < outputDim; x++) {
				for (int y = 0; y < outputDim; y++) {
					int x_stride = x * _stride;
					int y_stride = y * _stride;
					currentMaxIndexMatrix.transpose().block(x_stride, y_stride, 2, 2) *=
							getPreviousGradients()(i, y + x * outputDim + c * outputDim * outputDim);
				}
			}
			dX.block(i, imgSize * c, 1, imgSize) = currentMaxIndexMatrix.reshaped(1, imgSize);

		}

	}
	getInput()->setPreviousGradients(dX);

	/*
	 * Debug Information
	 */
	/*std::cout<<" MaxPoolOp FOrward:"<<getForward()<<std::endl;
	std::cout<<" MaxPoolOp Backwards:"<<getCurrentGradients()<<std::endl;*/
}


const Eigen::MatrixXf &MaxPoolOp::getMaxIndexMatrix() const {
	return _maxIndexMatrix;
}

void MaxPoolOp::setMaxIndexMatrix(const Eigen::MatrixXf &maxIndexMatrix) {
	_maxIndexMatrix = maxIndexMatrix;
}
