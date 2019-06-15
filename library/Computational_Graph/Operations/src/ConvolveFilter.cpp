//
// Created by phili on 14.06.2019.
//

#include "ConvolveFilter.hpp"

#include <iostream>


void ConvolveFilter::forwards() {
	//this results in a Vector containing in each row the result for a different input of the Batch
	//	setForward(getInputA()->getForward() * getInputB()->getForward());
	//TODO for efficency maybe change eigen to row major order for convfilter this should make sense, not sure about the
	// other operations ( as long as each input is a row)
	int imgN = getInputA()->getForward().rows();
	int filterN = getInputB()->getForward().rows();
	int imgDim = getInputA()->getForward().cols();
	int filterDim = getInputB()->getForward().cols();
	int outputDim = std::floor((imgDim - filterDim) / _stride) + 1;
//	int outDimSquare = std::pow(outputDim,2);
	Eigen::MatrixXf outputMatrix = Eigen::MatrixXf::Zero(imgN, outputDim * filterN);
	//loop over all images :
	for (int i = 0; i < imgN; i++) {
		//1. img= get each row of X; imgDim = img.dim
//		auto img =getInputA()->getForward()[i];
		//2. loop over all filters
		for (int j = 0; j < filterN; j++) {
			//2.1 filter = row of W; filterDim = filter.dim
//			auto filter = getInputB()->getForward()[j];
			//2.2 loop over outputDim
			for (int o = 0; o < outputDim; o++) {
				//2.2.1 part = img.block(...) <-- should get the part of the image the kernel is applied to
				/*std::cout<<"outputMatrix:"<<std::endl;
				std::cout<<outputMatrix<<std::endl;
				std::cout<<"A:"<<std::endl;
				std::cout<<getInputA()->getForward().block(i,
						o * filterDim + (_stride - 1) * o, 1, filterDim)<<std::endl;
				std::cout<<"B:"<<std::endl;
				std::cout<<getInputB()->getForward().block(j, 0, 1, filterDim)<<std::endl;*/
				outputMatrix(i, o + j * outputDim) =
						(getInputA()->getForward().block(i,
								o * filterDim + (_stride - 1) * o, 1, filterDim).cwiseProduct(
										getInputB()->getForward().block(j, 0, 1, filterDim)))
								.sum();

			}

		}

	}

	setForward(outputMatrix);


};

void ConvolveFilter::backwards() {

//	Eigen::MatrixXf inputGradient = getCurrentGradients() * (getInputB()->getForward().transpose());
//	Eigen::MatrixXf weightGradient = (getInputA()->getForward().transpose()) * getCurrentGradients();
//	getInputA()->setCurrentGradients(inputGradient);
//	getInputB()->setCurrentGradients(weightGradient);

}

std::string ConvolveFilter::printForward() {
	std::stringstream outStream;
	for (int i = 0; i < getForward().rows(); i++) {
		for (int j = 0; j < getForward().cols(); j++) {
			outStream << getForward()(i, j) << "\t";
		}
		outStream << std::endl;
	}
	return outStream.str();
}