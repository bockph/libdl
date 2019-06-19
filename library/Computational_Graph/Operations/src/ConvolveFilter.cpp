//
// Created by phili on 14.06.2019.
//

#include "ConvolveFilter.hpp"

#include <iostream>


void ConvolveFilter::forwards() {
    // Makes sure amount of channels of input and filter is the same
    auto A =getInputA()->getChannels();
    auto B = getInputB()->getChannels();
    assert(getInputA()->getChannels()==getInputB()->getChannels());

    //this results in a Vector containing in each row the result for a different input of the Batch
	//	setForward(getInputA()->getForward() * getInputB()->getForward());
	//TODO for efficency maybe change eigen to row major order for convfilter this should make sense, not sure about the
	// other operations ( as long as each input is a row)

	int imgN = getInputA()->getForward().rows();
	int filterN = getInputB()->getForward().rows();
	int imgDim=getInputA()->getOutputDim();
    int filterDim = getInputB()->getOutputDim();
	int outputDim = std::floor((imgDim - filterDim) / _stride) + 1;
    int imgSizeOneChannel = std::pow(imgDim,2);
    int filterSizeOneChannel = std::pow(filterDim,2);
	Eigen::MatrixXf outputMatrix = Eigen::MatrixXf::Zero(imgN, outputDim*outputDim * filterN);
	//loop over all images :
	for (int i = 0; i < imgN; i++) {
		for(int c =0;c<getInputA()->getChannels();c++){
            Eigen::MatrixXf tmpIMG = getInputA()->getForward().block(i,imgSizeOneChannel*c,1,imgSizeOneChannel);
            tmpIMG.resize(imgDim,imgDim);
            tmpIMG.transposeInPlace();
            //2. loop over all filters
            for (int j = 0; j < filterN; j++) {
                //get the Filter as Matrix, currently only with greyscale Images
                auto rowsFilter =getInputB()->getForward().rows();
                auto colsFilter =getInputB()->getForward().cols();
                Eigen::MatrixXf tmpFilter = getInputB()->getForward().block(j,filterSizeOneChannel*c,1,filterSizeOneChannel);
                tmpFilter.resize(filterDim,filterDim);
                tmpFilter.transposeInPlace();

                //2.2 loop over amount of possible convolutions and apply filter to img
                for (int x = 0; x < outputDim; x++) {
                    for(int y =0;y< outputDim; y++){
                        int x_stride = x*_stride;
                        int y_stride = y*_stride;
                        outputMatrix(i, y+x*outputDim+j*outputDim*outputDim) +=
                                tmpIMG.block(x_stride,y_stride,filterDim,filterDim).cwiseProduct(tmpFilter).sum();
                    }
                }

            }

        }

	}

	setForward(outputMatrix);
	setOutputDim(outputDim);
	setChannels(filterN);


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