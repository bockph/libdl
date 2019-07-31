//
// Created by pbo on 17.06.19.
//

#include <iostream>
#include "MaxPoolOp.hpp"

MaxPoolOp::MaxPoolOp(std::shared_ptr<Node> X, int windowDim, int stride)
		: NormalFunction(X, X->getOutputChannels()), _windowSize(windowDim), _stride(stride) {};

void MaxPoolOp::forwardPass() {
    /*
     * Calculate Sizes and Dimensions of miniBatch
     */
	int batchSize = static_cast<int>(getInput()->getForward().rows());
	int sampleSizeOneChannel = static_cast<int>(getInput()->getForward().cols() / getInputChannels());
	int sampleDim = static_cast<int>(std::sqrt(sampleSizeOneChannel));
	int outputDim = static_cast<int>(std::floor((sampleDim - _windowSize) / _stride) + 1);

	Eigen::MatrixXf::Index maxRow, maxCol;


	Eigen::MatrixXf outputMatrix = Eigen::MatrixXf::Zero(batchSize, outputDim * outputDim * getOutputChannels());
	Eigen::MatrixXf maxIndexMatrix = Eigen::MatrixXf::Zero(batchSize, getInput()->getForward().cols());


	//loop over all samples :
	for (int i = 0; i < batchSize; i++) {
		//loop over all channels
		for (int c = 0; c < getInputChannels(); c++) {
			//get the Current Channel of the Current Sample
			Eigen::MatrixXf currentChannelOfSample = getInput()->getForward().block(i, sampleSizeOneChannel * c, 1, sampleSizeOneChannel)
					.reshaped(sampleDim, sampleDim);

			//get the current block of the maxIndexMatrix
			Eigen::MatrixXf currentMaxIndexMatrix = maxIndexMatrix.block(i,
                                                                         sampleSizeOneChannel * c, 1, sampleSizeOneChannel).reshaped(sampleDim, sampleDim);


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
                                 sampleSizeOneChannel * c, 1, sampleSizeOneChannel) = currentMaxIndexMatrix.reshaped(1, sampleSizeOneChannel).block(0, 0, 1, sampleSizeOneChannel);

		}


	}
	setMaxIndexMatrix(maxIndexMatrix);
	setForward(outputMatrix);
}

void MaxPoolOp::backwardPass() {

	int batchSize = static_cast<int>(getInput()->getForward().rows());
	int sampleSizeOneChannel = static_cast<int>(getInput()->getForward().cols() / getInputChannels());
	int sampleDim = static_cast<int>(std::sqrt(sampleSizeOneChannel));
	int outputDim = static_cast<int>(std::floor((sampleDim - _windowSize) / _stride) + 1);


	Eigen::MatrixXf dX = Eigen::MatrixXf::Zero(batchSize, sampleSizeOneChannel * getInputChannels());


	//loop over all samples :
	for (int i = 0; i < batchSize; i++) {
		//loop over all channels
		for (int c = 0; c < getInputChannels(); c++) {
			//get the maxIndexMatrix for current sample & channel
			Eigen::MatrixXf currentMaxIndexMatrix = getMaxIndexMatrix()
					.block(i, sampleSizeOneChannel * c, 1, sampleSizeOneChannel).reshaped(sampleDim, sampleDim);
			//apply Gradients to corrresponding maxIndex
			for (int x = 0; x < outputDim; x++) {
				for (int y = 0; y < outputDim; y++) {
					int x_stride = x * _stride;
					int y_stride = y * _stride;
					currentMaxIndexMatrix.transpose().block(x_stride, y_stride, 2, 2) *=
							getPreviousGradients()(i, y + x * outputDim + c * outputDim * outputDim);
				}
			}
			dX.block(i, sampleSizeOneChannel * c, 1, sampleSizeOneChannel) = currentMaxIndexMatrix.reshaped(1, sampleSizeOneChannel);

		}

	}
	getInput()->setPreviousGradients(dX);
}


const Eigen::MatrixXf &MaxPoolOp::getMaxIndexMatrix() const {
	return _maxIndexMatrix;
}

void MaxPoolOp::setMaxIndexMatrix(const Eigen::MatrixXf &maxIndexMatrix) {
	_maxIndexMatrix = maxIndexMatrix;
}
