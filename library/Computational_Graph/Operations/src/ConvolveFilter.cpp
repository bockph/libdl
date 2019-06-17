//
// Created by phili on 14.06.2019.
//

#include "ConvolveFilter.hpp"

#include <iostream>

int getChannels(){
    return 1;
}
void ConvolveFilter::forwards() {
	//this results in a Vector containing in each row the result for a different input of the Batch
	//	setForward(getInputA()->getForward() * getInputB()->getForward());
	//TODO for efficency maybe change eigen to row major order for convfilter this should make sense, not sure about the
	// other operations ( as long as each input is a row)
	int imgN = getInputA()->getForward().rows();
	int filterN = getInputB()->getForward().rows();
	int imgDim = getInputA()->getForward().cols();
	int imgDimSQRT=std::sqrt(imgDim);

    int amountPixels = imgDim/getChannels();
    int amountRows = std::sqrt(amountPixels);
    int amountCols = amountRows*getChannels();


    int filterDim = getInputB()->getForward().cols();
    int filterDimSQRT = std::sqrt(filterDim);
	int outputDimSQRT = std::floor((imgDimSQRT - filterDimSQRT) / _stride) + 1;
	int outputDim = std::pow(outputDimSQRT,2);
	Eigen::MatrixXf outputMatrix = Eigen::MatrixXf::Zero(imgN, outputDim * filterN);
	//loop over all images :
	for (int i = 0; i < imgN; i++) {
		//get the Image as Matrix, currently only with greyscale/oneChannel Images
		//TODO: Implement multi channels
        Eigen::MatrixXf tmpIMG = getInputA()->getForward().block(i,0,1,imgDim);
        tmpIMG.resize(imgDimSQRT,imgDimSQRT);
        tmpIMG.transposeInPlace();
		//2. loop over all filters
		for (int j = 0; j < filterN; j++) {
            //get the Filter as Matrix, currently only with greyscale Images
            //TODO: Implement multi channels
            Eigen::MatrixXf tmpFilter = getInputB()->getForward().block(j,0,1,filterDim);
            tmpFilter.resize(filterDimSQRT,filterDimSQRT);
            tmpFilter.transposeInPlace();

            std::cout<<"Filter:"<<tmpFilter<<std::endl;
			//2.2 loop over amount of possible convolutions and apply filter to img
            for (int x = 0; x < outputDimSQRT; x++) {
                for(int y =0;y< outputDimSQRT; y++){
                    int x_stride = x*_stride;
                    int y_stride = y*_stride;
                    outputMatrix(i, y+x*outputDimSQRT) =
                            tmpIMG.block(x_stride,y_stride,filterDimSQRT,filterDimSQRT).cwiseProduct(tmpFilter).sum();
                }
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