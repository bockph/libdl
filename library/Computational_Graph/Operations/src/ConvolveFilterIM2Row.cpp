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





Eigen::MatrixXf ConvolveFilterIM2COL::im2col(const Eigen::MatrixXf &input, const Eigen::MatrixXf &filter, int stride, int channel, int batchSize) {



	int filterSize = filter.cols() / channel;
	int filterDim = std::sqrt(filterSize);

	int inputSize = input.cols() / channel;
	int inputDim = std::sqrt(inputSize);
	int outputDim = std::floor((inputDim - filterDim) / stride) + 1;
	int outputSize = std::pow(outputDim, 2);

	Eigen::MatrixXf inputTransformed(filter.cols(), outputSize*batchSize);
    for (int s = 0; s < batchSize; s++) {

        for (int c = 0; c < channel; c++) {
            Eigen::MatrixXf tmp = input.block(s, inputSize * c, 1, inputSize);
            tmp.resize(inputDim, inputDim);

            for (int i = 0; i + filterDim <= inputDim; i += stride) {
                for (int j = 0; j + filterDim <= inputDim; j += stride) {
                    Eigen::MatrixXf col = tmp.block(i, j, filterDim, filterDim);
                    col.resize(filterSize, 1);
                    inputTransformed.block(c * filterSize, s*outputSize+i / stride + j / stride * outputDim, filterSize, 1) = col;
                }
            }
        }
    }


	return inputTransformed;

}

Eigen::MatrixXf ConvolveFilterIM2COL::col2im(const Matrix &input, const Eigen::MatrixXf &filter,int origDim, int stride, int channel,int batchSize) {
    int filterSize = filter.cols() / channel;
    int filterDim = std::sqrt(filterSize);

    int inputSize = input.cols();
    int inputDim = std::sqrt(inputSize);
    int outputDim = origDim;//(inputDim-1)*stride+filterDim;//std::floor((inputDim - filterDim) / stride) + 1;
    int outputSize = std::pow(outputDim, 2);

    Eigen::MatrixXf inputTransformed(batchSize,outputSize*channel);

        for (int c = 0; c < channel; c++) {
            int collumn=0;
            for (int s = 0; s < batchSize; s++) {

            Eigen::MatrixXf tmp= Eigen::MatrixXf::Zero(outputDim,outputDim);
            /*int collumn=0;
            for(int i=0;i<outputDim;i++){
                for(int j=0;j<outputDim;j++){
                    tmp.block(j*stride,i*stride,filterSize,filterSize)+=input.block(c*filterSize,s*outputSize+collumn,filterSize,1).reshaped(filterDim,filterDim);
                    collumn++;
                }
            }
            inputTransformed.block(s,c*outputSize,1,outputSize)=tmp.reshaped(1,outputSize);*/


            for (int i = 0; i + filterDim <= outputDim; i += stride) {
                for (int j = 0; j + filterDim <= outputDim; j += stride) {
                    Eigen::MatrixXf col = input.block(c * filterSize, collumn,filterSize,1);//s*outputSize+i / stride + j / stride * inputDim, filterSize, 1);
                    col.resize(filterDim,filterDim);
                    tmp.block(i, j, filterDim, filterDim)+=col;
                    collumn++;
                }
            }
            tmp.resize(1,outputSize);
            inputTransformed.block(s,c*outputSize,1,outputSize)=tmp;
        }
    }



    return inputTransformed;

}

/*void forwardConvolution(Matrix outputMatrix, Eigen::MatrixXf& im2ColMatrix, Eigen::MatrixXf& ){
    for (int i = 0; i < getAmountOfInputs(); i++) {
        Eigen::MatrixXf sample = getInputA()->getForward().row(i);
        setIm2Col(im2col(sample, getInputB()->getForward(), _stride, getInputChannels()));
        Eigen::MatrixXf conv = getInputB()->getForward() * getIm2Col() ;

        conv.transposeInPlace();
        conv.resize(1, conv.size());
        outputMatrix.block(i, 0, 1, getOutputSize()) = conv;
    }
}*/


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


	Eigen::MatrixXf outputMatrix = Eigen::MatrixXf::Zero(getAmountOfInputs(), getOutputSize());
	//loop over all images :
	startTimeMeasurement();


        setIm2Col(im2col(getInputA()->getForward(), getInputB()->getForward(), _stride, getInputChannels(),getAmountOfInputs()));

        Eigen::MatrixXf conv = getInputB()->getForward() * getIm2Col();

    //reshape convolution to outputForm
    //TODO change outputForm
    int batchSize=getAmountOfInputs();
    int size =getOutputSizePerChannel();
    int channel = getOutputChannels();
    Matrix output(batchSize,size*channel);
    for(int i = 0;i<channel;i++){
        Matrix tmp = conv.row(i);//block(0,i*size,channel,size);
        tmp=tmp.reshaped<Eigen::RowMajor>(batchSize,size).eval();
        outputMatrix.block(0,i*size,batchSize,size)=tmp;
    }


	stopTimeMeasurement(0);

    setForward(outputMatrix);



}


void ConvolveFilterIM2COL::backwards() {
	startTimeMeasurement();

//Gradient of Kernel is convolution of inputA with the gradient


        int outputDim=getOutputSizePerChannel();
        int outputChannel=getInputB()->getForward().rows();
        int batchSize=getAmountOfInputs();

    //reshape gradients
        int col = batchSize*outputDim;
        int row = outputChannel;
        Eigen::MatrixXf gradtmp=Matrix::Zero(row,col);

        for(int i = 0;i<batchSize;i++){
            Matrix tmp = getCurrentGradients().row(i);//block(0,i*size,channel,size);
            tmp=tmp.reshaped<Eigen::RowMajor>(outputChannel,outputDim).eval();
            gradtmp.block(0,i*outputDim,outputChannel,outputDim)=tmp;
        }

        Eigen::MatrixXf gradientX = getInputB()->getForward().transpose()*gradtmp;

        gradientX=col2im(gradientX,getInputB()->getForward(),getInputDimX(),_stride,getInputChannels(),getAmountOfInputs());


    Eigen::MatrixXf gradientW  = gradtmp*getIm2Col().transpose();

	getInputB()->setCurrentGradients(gradientW);
	getInputA()->setCurrentGradients(gradientX);

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
