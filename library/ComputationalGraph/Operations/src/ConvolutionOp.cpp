//
// Created by phili on 14.06.2019.
//

#include <iostream>
#include "ConvolutionOp.hpp"


ConvolutionOp::ConvolutionOp(std::shared_ptr<Node> X, std::shared_ptr<Parameter> W, int stride)
		: NormalFunction(X, W, static_cast<int>(W->getForward().rows())), _stride(stride) {
	if (getInputChannels() != W->getOutputChannels()) {
		throw std::invalid_argument("Input X and W should have the same amount of Channels");
	}
}


void ConvolutionOp::im2col(Matrix &output, const Matrix &input, int filterSize, int stride, int channel,
						   int batchSize) {

	int filterDim = static_cast<int>(std::sqrt(filterSize));

	int inputSize = static_cast<int>(input.cols() / channel);
	int inputDim = static_cast<int>(std::sqrt(inputSize));
	int outputDim = static_cast<int>(std::floor((inputDim - filterDim) / stride) + 1);
	int outputSize = static_cast<int>(std::pow(outputDim, 2));

	output = Matrix(filterSize * channel, outputSize * batchSize);

	for (int c = 0; c < channel; c++) {
		int collumn = 0;
		for (int s = 0; s < batchSize; s++) {
			//TODO different naming
			Matrix tmp = input.block(s, inputSize * c, 1, inputSize).reshaped(inputDim, inputDim);

			for (int i = 0; i + filterDim <= inputDim; i += stride) {
				for (int j = 0; j + filterDim <= inputDim; j += stride) {
					//reshaped leads to an Transpose, thats why we take block(j,i,...) instead of block(i,j),
					// -->TODO  check if row wise reshape makes sense
					Matrix col = tmp.block(j, i, filterDim, filterDim).reshaped(filterSize, 1);
					output.block(c * filterSize, collumn, filterSize, 1) = col;
					collumn++;
				}
			}
		}
	}

}

void ConvolutionOp::col2im(Matrix &output, const Matrix &input, int filterSize, int outputDim, int stride, int channel,
						   int batchSize) {
	int filterDim = static_cast<int>(std::sqrt(filterSize));
	int outputSize = static_cast<int>(std::pow(outputDim, 2));

	output = Matrix(batchSize, outputSize * channel);

	for (int c = 0; c < channel; c++) {
		int collumn = 0;
		for (int s = 0; s < batchSize; s++) {

			Matrix tmp = Matrix::Zero(outputDim, outputDim);
			for (int i = 0; i + filterDim <= outputDim; i += stride) {
				for (int j = 0; j + filterDim <= outputDim; j += stride) {
					Matrix col = input.block(
							c * filterSize, collumn, filterSize, 1).reshaped(filterDim, filterDim);
					//reshaped leads to an Transpose, thats why we take block(j,i,...) instead of block(i,j),
					// -->TODO  check if row wise reshape makes sense
					tmp.block(j, i, filterDim, filterDim) += col;
					collumn++;
				}
			}
			output.block(s, c * outputSize, 1, outputSize) = tmp.reshaped(1, outputSize);
		}
	}

}

void ConvolutionOp::forwardsConvolution(const Matrix &miniBatch, const Matrix &filter, Matrix &im2ColM,
										Matrix &outputMatrix, const int stride, const int channels) {
	int amountSamples = static_cast<int>(miniBatch.rows());
	int sampleSizeOneChannel = static_cast<int>(miniBatch.cols() / channels);
	int sampleDim = static_cast<int>(std::sqrt(sampleSizeOneChannel));

	int amountFilters = static_cast<int>(filter.rows());
	int kernelSize = static_cast<int>(filter.cols());
	int kernelSizeOneChannel = kernelSize / channels;
	int kernelDim = static_cast<int>(std::sqrt(kernelSizeOneChannel));

	int outputDim = static_cast<int>(std::floor(sampleDim - kernelDim) / stride + 1);
	int outputSize = static_cast<int>(std::pow(outputDim, 2));

	im2col(im2ColM, miniBatch, kernelSizeOneChannel, stride, channels, amountSamples);

	Matrix conv = filter * im2ColM;

//Reshape extract method
	//TODO change outputForm
	outputMatrix = Matrix::Zero(amountSamples, outputSize * amountFilters);

	for (int i = 0; i < amountSamples; i++) {
		for (int f = 0; f < amountFilters; f++) {
			outputMatrix.block(i, outputSize * f, 1, outputSize) = conv.block(f, i * outputSize, 1, outputSize);
		}
	}


}


void ConvolutionOp::forwardPass() {

	Matrix im2ColM, outputMatrix;


	forwardsConvolution(getInput()->getForward(), getParameter()->getForward(), im2ColM, outputMatrix, _stride,
			getInputChannels());


	setIm2Col(im2ColM);

	setForward(outputMatrix);


}

void
ConvolutionOp::backwardsConvolution(Matrix &dX, Matrix &dW, const Matrix &filter, const Matrix &dout,
									const Matrix &im2ColM, int batchSize, int inputDimX, int stride,
									int channelsX) {

	int amountFilter = static_cast<int>(filter.rows());
	int outputSize = static_cast<int>(dout.cols() / amountFilter);
	int filterSizeOneChannel = static_cast<int>(filter.cols() / channelsX);


	Matrix doutReshaped = Matrix::Zero(amountFilter, batchSize * outputSize);
	for (int i = 0; i < batchSize; i++) {
		for (int f = 0; f < amountFilter; f++) {
			doutReshaped.block(f, i * outputSize, 1, outputSize) = dout.block(i, outputSize * f, 1, outputSize);
		}
	}

	Matrix dXBeforeReshape = filter.transpose() * doutReshaped;

	col2im(dX, dXBeforeReshape, filterSizeOneChannel, inputDimX, stride, channelsX, batchSize);

	dW = doutReshaped * im2ColM.transpose();

}

void ConvolutionOp::backwardPass() {


	int batchSize = static_cast<int>(getInput()->getForward().rows());
	int inputDimX = static_cast<int>(std::sqrt(getInput()->getForward().cols() / getInputChannels()));

	Matrix gradientX, gradientW;

	backwardsConvolution(gradientX, gradientW, getParameter()->getForward(), getPreviousGradients(), getIm2Col(), batchSize,
			inputDimX, _stride, getInputChannels());


	getParameter()->setPreviousGradients(gradientW);
	getInput()->setPreviousGradients(gradientX);
}


const Matrix &ConvolutionOp::getIm2Col() const {
	return _im2Col;
}

void ConvolutionOp::setIm2Col(const Matrix &im2Col) {
	_im2Col = im2Col;
}
