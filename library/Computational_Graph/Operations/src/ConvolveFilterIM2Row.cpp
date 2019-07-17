//
// Created by phili on 14.06.2019.
//

#include <iostream>
#include "ConvolveFilterIM2COL.hpp"

std::time_t compute_ConvolveFilter;
std::time_t backwards_ConvolveFilter;

//THis need to be moved to the forward class
ConvolveFilterIM2COL::ConvolveFilterIM2COL(std::shared_ptr<Node> X, std::shared_ptr<Filter> W, int stride)
		:
		Operation(X, W), _stride(stride), _amountFilters(W->getForward().rows()) {

	setOutputDim(std::floor((getInputDimX() - getInputDimW()) / _stride) + 1);
	setOutputChannels(getAmountFilters());
	setImgSizeOneChannel(std::pow(getInputDimX(), 2));
	setFilterSizeOneChannel(std::pow(getInputDimW(), 2));
	setOutputSizeOneFilter(std::pow(getOutputDim(), 2));
	setOutputSize(getOutputSizePerChannel() * getOutputChannels());
}

void ConvolveFilterIM2COL::addPadding(Eigen::MatrixXf &m, int rowPadding, int colPadding) {

	Eigen::MatrixXf paddedMatrix = Eigen::MatrixXf::Zero(m.rows() + rowPadding * 2, m.cols() + colPadding * 2);
	paddedMatrix.block(rowPadding, colPadding, m.rows(), m.cols()) = m;
	m = paddedMatrix;
}





Eigen::MatrixXf ConvolveFilterIM2COL::im2col(Eigen::MatrixXf &input, const Eigen::MatrixXf &filter, int stride, int channel) {
	int filterSize = filter.cols() / channel;
	int filterDim = std::sqrt(filterSize);

	int inputSize = input.cols() / channel;
	int inputDim = std::sqrt(inputSize);
	int outputDim = std::floor((inputDim - filterDim) / stride) + 1;
	int outputSize = std::pow(outputDim, 2);

	Eigen::MatrixXf inputTransformed(filter.cols(), outputSize);

	for (int c = 0; c < channel; c++) {
		Eigen::MatrixXf tmp = input.block(0, inputSize * c, 1, inputSize);
        tmp.resize(inputDim, inputDim);

        for (int i = 0; i + filterDim <= inputDim; i += stride) {
			for (int j = 0; j + filterDim <= inputDim; j += stride) {
				Eigen::MatrixXf col = tmp.block(i, j, filterDim, filterDim);
				col.resize(filterSize, 1);
                inputTransformed.block(c * filterSize, i / stride + j / stride * outputDim, filterSize, 1) = col;
			}
		}
	}

//    setIm2Col(inputTransformed);

	return inputTransformed;

}

Eigen::MatrixXf ConvolveFilterIM2COL::col2im(Eigen::MatrixXf &input, const Eigen::MatrixXf &filter,int origDim, int stride, int channel) {
    int filterSize = filter.cols() / channel;
    int filterDim = std::sqrt(filterSize);

    int inputSize = input.cols();
    int inputDim = std::sqrt(inputSize);



    int outputDim = origDim;//(inputDim-1)*stride+filterDim;//std::floor((inputDim - filterDim) / stride) + 1;
    int outputSize = std::pow(outputDim, 2);

    Eigen::MatrixXf inputTransformed(1,outputSize*channel);


    for (int c = 0; c < channel; c++) {
        Eigen::MatrixXf tmp= Eigen::MatrixXf::Zero(outputDim,outputDim);
//        Eigen::MatrixXf tmp = input.block(c*filterSize, 0,filterSize,input.cols());
//        tmp.resize(inputDim, inputDim);

        for (int i = 0; i + filterDim <= outputDim; i += stride) {
            for (int j = 0; j + filterDim <= outputDim; j += stride) {
                Eigen::MatrixXf col = input.block(c * filterSize, i / stride + j / stride * inputDim, filterSize, 1);
                col.resize(filterDim,filterDim);
                tmp.block(i, j, filterDim, filterDim)+=col;

//                inputTransformed.block(c * filterSize, i / stride + j / stride * outputDim, filterSize, 1) = col;
            }
        }
        tmp.resize(1,outputSize);
        inputTransformed.block(0,c*outputSize,1,outputSize)=tmp;

    }

//    setIm2Col(inputTransformed);

    return inputTransformed;

}


void ConvolveFilterIM2COL::forwards() {
//    std::cout<<"im2cols"<<std::endl;


/*
	* GENERALL STUFF
	*/
	//    setOutputChannels(getInputA()->getOutputChannels());
	setAmountOfInputs(getInputA()->getForward().rows());
	setOutputDim(std::floor((getInputDimX() - getInputDimW()) / _stride) + 1);
	setOutputChannels(getAmountFilters());
	setImgSizeOneChannel(std::pow(getInputDimX(), 2));
	setFilterSizeOneChannel(std::pow(getInputDimW(), 2));
	setOutputSizeOneFilter(std::pow(getOutputDim(), 2));
	setOutputSize(getOutputSizePerChannel() * getOutputChannels());
	/*
	 *
	 */

	//TODO for efficency maybe change eigen to row major order for convfilter this should make sense, not sure about the
	// other operations ( as long as each input is a row)

	double convolutionCounter = 0;
//    std::cout<<"GOOOOOOOOOOOOOOOOOOOOOOOOOOO"<<std::endl;
	Eigen::MatrixXf outputMatrix = Eigen::MatrixXf::Zero(getAmountOfInputs(), getOutputSize());
	//loop over all images :
	startTimeMeasurement();
//#pragma parallel for num_threads(8)

	for (int i = 0; i < getAmountOfInputs(); i++) {
		Eigen::MatrixXf sample = getInputA()->getForward().row(i);
        setIm2Col(im2col(sample, getInputB()->getForward(), _stride, getInputChannels()));
        Eigen::MatrixXf conv = getInputB()->getForward() * getIm2Col() ;

        conv.transposeInPlace();
        conv.resize(1, conv.size());
		outputMatrix.block(i, 0, 1, getOutputSize()) = conv;
	}
	stopTimeMeasurement(0);

	setForward(outputMatrix);



}

void ConvolveFilterIM2COL::backwards() {
	startTimeMeasurement();

//Gradient of Kernel is convolution of inputA with the gradient
	Eigen::MatrixXf gradientsKernel = Eigen::MatrixXf::Zero(getAmountFilters(),
			_filterSizeOneChannel * getInputChannels());
	Eigen::MatrixXf gradientsInput = Eigen::MatrixXf::Zero(getAmountOfInputs(),
			getImgSizeOneChannel() * getInputChannels());

    Eigen::MatrixXf filterT = getInputB()->getForward().transpose();

	/*
		 * For X :
		 * 1. get Current Gradient and apply padding and stride padding to each channel, channel = amountFilters
		 * 2. convolve with filter to according channel and use reduce_sum to make one row
		 * |inputchannel| =|initialfilters| , filter = input B reversed whereas channel and filteramount switch
		 * doDilationPerChannel()
		 * doPaddingPerChannel()
		 *                      currentInputB.reverse().eval();

		 * im2cOL(dilAndPadedGradient,getInputB()->getForward(), _stride?,getAmountOfFilters());
		 *
		 * improve reverse of filter before loop
		 * For W:
		 * 1.get Curent Gradient and apply only stride padding to each channel
		 * 2. |channel| * convolve the cGradient over A
		 *
		 * doDilationPerChannel()
		 *
		 * im2col(currentInputA,currentGradient,
		 *
		 * every gradient channel contributes to one filter, and every inputA channel contributes to one filter channel
		 */


    //loop over all images :
//#pragma parallel for num_threads(8)

    for (int i = 0; i < getAmountOfInputs(); i++) {
        //DO convolution for image i



        Eigen::MatrixXf gradtmp(getInputB()->getForward().rows(),getIm2Col().cols() );//=getCurrentGradients().row(i);
    // TODO use reshape instead of loop
      for(int r=0;r<getInputB()->getForward().rows();r++){
            gradtmp.block(r,0,1,getIm2Col().cols())=getCurrentGradients().row(i).block(0,r*getIm2Col().cols(),1,getIm2Col().cols());
        }

//        std::cout<<"gradientsResize:\n"<<gradtmp<<std::endl;
        Eigen::MatrixXf tmp = filterT*gradtmp;
        tmp=col2im(tmp,getInputB()->getForward(),getInputDimX(),_stride,getInputChannels());

        gradientsInput.block(i,0,1,_imgSizeOneChannel*getInputChannels())=tmp;

        Eigen::MatrixXf tmpW = gradtmp*getIm2Col().transpose();
        gradientsKernel += tmpW;


    }

	getInputB()->setCurrentGradients(gradientsKernel);
	getInputA()->setCurrentGradients(gradientsInput);

	stopTimeMeasurement(1);


}

std::string ConvolveFilterIM2COL::printForward() {
	std::stringstream outStream;
	for (int i = 0; i < getForward().rows(); i++) {
		for (int j = 0; j < getForward().cols(); j++) {
			outStream << getForward()(i, j) << "\t";
		}
		outStream << std::endl;
	}
	return outStream.str();
}


int ConvolveFilterIM2COL::getAmountFilters() const {
	return _amountFilters;
}



int ConvolveFilterIM2COL::getImgSizeOneChannel() const {
	return _imgSizeOneChannel;
}

void ConvolveFilterIM2COL::setImgSizeOneChannel(int imgSizeOneChannel) {
	_imgSizeOneChannel = imgSizeOneChannel;
}


void ConvolveFilterIM2COL::setFilterSizeOneChannel(int filterSizeOneChannel) {
	_filterSizeOneChannel = filterSizeOneChannel;
}

int ConvolveFilterIM2COL::getOutputSizePerChannel() const {
	return _outputSizeOneFilter;
}

void ConvolveFilterIM2COL::setOutputSizeOneFilter(int outputSizeOneFilter) {
	_outputSizeOneFilter = outputSizeOneFilter;
}

int ConvolveFilterIM2COL::getOutputSize() const {
	return _outputSize;
}

void ConvolveFilterIM2COL::setOutputSize(int outputSize) {
	_outputSize = outputSize;
}

const Eigen::MatrixXf &ConvolveFilterIM2COL::getIm2Col() const {
    return _im2Col;
}

void ConvolveFilterIM2COL::setIm2Col(const Eigen::MatrixXf &im2Col) {
    _im2Col = im2Col;
}
