//
// Created by pbo on 17.06.19.
//

#include <iostream>
#include "MaxPool.hpp"

int getChannels() {
	return 1;
}

void MaxPool::forwards() {

	/*
	* GENERALL STUFF
	*/
	//    setOutputChannels(getInputA()->getOutputChannels());
	beforeForward();
	/*
	 *
	 */
	//this results in a Vector containing in each row the result for a different input of the Batch
	//	setForward(getInputA()->getForward() * getInputB()->getForward());
	//TODO for efficency maybe change eigen to row major order for convfilter this should make sense, not sure about the
	// other operations ( as long as each input is a row)

	int imgN = getInputA()->getForward().rows();
	int imgDim = getInputA()->getOutputDim();
	int outputDim = std::floor((imgDim - _windowSize) / _stride) + 1;
	int imgSizeOneChannel = std::pow(imgDim, 2);


	Eigen::MatrixXf::Index maxRow, maxCol;


	Eigen::MatrixXf outputMatrix = Eigen::MatrixXf::Zero(imgN, outputDim * outputDim * getOutputChannels());
	Eigen::MatrixXf indexMatrix = Eigen::MatrixXf::Zero(imgN, getInputA()->getForward().cols()); //, imgDim * imgDim);

	//loop over all images :
	for (int i = 0; i < imgN; i++) {

		for (int c = 0; c < getInputA()->getOutputChannels(); c++) {
			Eigen::MatrixXf tmpIMG = getInputA()->getForward().block(i, imgSizeOneChannel * c, 1, imgSizeOneChannel);

			tmpIMG.resize(imgDim, imgDim);
			tmpIMG.transposeInPlace();
			Eigen::MatrixXf indexTMP = indexMatrix.block(i, imgSizeOneChannel * c, 1, imgSizeOneChannel);
			indexTMP.resize(imgDim, imgDim);
			indexTMP.transposeInPlace();
//			Eigen::MatrixXf indexTMP = Eigen::MatrixXf::Zero(imgDim, imgDim);




			//2.2 loop over amount of possible convolutions and apply filter to img
			for (int x = 0; x < outputDim; x++) {
				for (int y = 0; y < outputDim; y++) {
					int x_stride = x * _stride;
					int y_stride = y * _stride;
					outputMatrix(i, y + x * outputDim + c * outputDim * outputDim) =
							tmpIMG.block(x_stride, y_stride, 2, 2).maxCoeff(&maxRow, &maxCol);
//					indexTMP.block(x_stride, y_stride, 2, 2)(i,
//							maxCol + maxRow * outputDim + c * outputDim * outputDim) = 1;
//				auto inTmp =indexTMP.block(i, imgSizeOneChannel * c, 1, imgSizeOneChannel);
					indexTMP.block(x_stride, y_stride, 2, 2)
							(maxRow, maxCol) = 1;

				}
			}
			indexTMP.transposeInPlace();
			indexTMP.resize(1, imgSizeOneChannel);
			indexMatrix.block(i, imgSizeOneChannel * c, 1, imgSizeOneChannel) = indexTMP.block(0, 0, 1, imgSizeOneChannel);

		}


	}
	setMaxIndexMatrix(indexMatrix);
	setForward(outputMatrix);
   /* Eigen::MatrixXf test = getForward();
//    test.transpose().eval();
    test.resize(getOutputDim(),getOutputDim()*getOutputChannels());
    std::cout<<"Output MAXPOOL:\n"<< getForward()<<std::endl;
    std::cout<<"\nOUTPUT MAXPOOL:\n"<<test<<std::endl;
    std::cout<<"Output Max:\n"<< getForward().array().maxCoeff()<<std::endl;
    std::cout<<"Output Average:\n"<< getForward().mean()<<std::endl;
    int x=2+3;*/

};

void MaxPool::backwards() {
	int imgN = getInputA()->getForward().rows();
	int imgDim = getInputA()->getOutputDim();
	int outputDim = std::floor((imgDim - _windowSize) / _stride) + 1;
	int imgSizeOneChannel = std::pow(imgDim, 2);


	Eigen::MatrixXf::Index maxRow, maxCol;


	Eigen::MatrixXf indexMatrix = Eigen::MatrixXf::Zero(imgN, imgSizeOneChannel*getInputA()->getOutputChannels());


	//loop over all images :
	for (int i = 0; i < imgN; i++) {
		for (int c = 0; c < getInputA()->getOutputChannels(); c++) {
			Eigen::MatrixXf tmpIMG = getMaxIndexMatrix().block(i, imgSizeOneChannel * c, 1, imgSizeOneChannel);
			tmpIMG.resize(imgDim, imgDim);
			tmpIMG.transposeInPlace();

			//2.2 loop over amount of possible convolutions and apply filter to img
			for (int x = 0; x < outputDim; x++) {
				for (int y = 0; y < outputDim; y++) {
					int x_stride = x * _stride;
					int y_stride = y * _stride;
					tmpIMG.block(x_stride, y_stride, 2, 2) *= getCurrentGradients()(i,
							y + x * outputDim + c * outputDim * outputDim);

				}
			}

			tmpIMG.transposeInPlace();
			tmpIMG.resize(1, imgDim * imgDim);
			indexMatrix.block(i, imgSizeOneChannel * c, 1, imgSizeOneChannel) = tmpIMG; //.block(0, 0, 1,
					//imgSizeOneChannel);

		}

	}
	getInputA()->setCurrentGradients(indexMatrix);


}

std::string MaxPool::printForward() {
	std::stringstream outStream;
	for (int i = 0; i < getForward().rows(); i++) {
		for (int j = 0; j < getForward().cols(); j++) {
			outStream << getForward()(i, j) << "\t";
		}
		outStream << std::endl;
	}
	return outStream.str();
}

const Eigen::MatrixXf &MaxPool::getMaxIndexMatrix() const {
	return _maxIndexMatrix;
}

void MaxPool::setMaxIndexMatrix(const Eigen::MatrixXf &maxIndexMatrix) {
	_maxIndexMatrix = maxIndexMatrix;
}
