//
// Created by phili on 14.06.2019.
//

#include <iostream>
#include "ConvolveFilter.hpp"

std::time_t compute_ConvolveFilter ;
std::time_t backwards_ConvolveFilter ;
//THis need to be moved to the forward class
ConvolveFilter::ConvolveFilter(std::shared_ptr<Node> X, std::shared_ptr<Filter> W,int stride ):
Operation(X, W),
_stride(stride),
_amountFilters(W->getForward().rows())
{

    setOutputDim(std::floor((getInputDimX() - getInputDimW()) / _stride) + 1);
    setOutputChannels(getAmountFilters());
    setImgSizeOneChannel(std::pow(getInputDimX(),2));
    setFilterSizeOneChannel(std::pow(getInputDimW(),2));
    setOutputSizeOneFilter(std::pow(getOutputDim(),2));
    setOutputSize( getOutputSizePerChannel() * getOutputChannels());
}

void ConvolveFilter::addPadding(Eigen::MatrixXf& m, int rowPadding, int colPadding){
    Eigen::MatrixXf paddedMatrix = Eigen::MatrixXf::Zero(m.rows()+rowPadding*2,m.cols()+colPadding*2);
    paddedMatrix.block(rowPadding,colPadding,m.rows(),m.cols())=m;
    m= paddedMatrix;
}
//TODO hier ungewollte transponierung?
void ConvolveFilter::addStrideDilation(Eigen::MatrixXf& m, int stride){
    if(stride==1) return;
//    stride-=1;
    int pRows = m.rows()+(stride-1)*(m.rows()-1) ;
    int pCols = m.cols()+(stride-1)*(m.cols()-1);
    Eigen::MatrixXf paddedMatrix = Eigen::MatrixXf::Zero(pRows,pCols);
    for(int i =0,  x=0;i<paddedMatrix.rows();i+=stride,x++){
        for(int j =0,  y=0;j<paddedMatrix.cols();j+=stride,y++){
            paddedMatrix(i,j)=m(x,y);
        }
    }
    m=paddedMatrix;
//    return paddedMatrix;
}

//TODO hier ungewollte transponierung? durch optimierung
//convolve is optimized for column-major order
Eigen::MatrixXf ConvolveFilter::convolve(const Eigen::MatrixXf &input, const Eigen::MatrixXf &kernel,int stride,int outputDim) {
	outputDim =std::floor((input.rows() - kernel.rows()) / stride) + 1;
   int kernelDim = kernel.rows();
   Eigen::MatrixXf output=Eigen::MatrixXf::Zero(outputDim,outputDim);

   for (int x = 0; x < outputDim; x++) {
       for(int y =0;y< outputDim; y++){
           int x_stride = x*stride;
           int y_stride = y*stride;
           output(y, x) += input.block(x_stride,y_stride,kernelDim,kernelDim).cwiseProduct(kernel).sum();
       }
   }
   return output;



}
void ConvolveFilter::forwards() {
    startTimeMeasurement();


/*
	* GENERALL STUFF
	*/
	//    setOutputChannels(getInputA()->getOutputChannels());
	setAmountOfInputs(getInputA()->getForward().rows());
	setOutputDim(std::floor((getInputDimX() - getInputDimW()) / _stride) + 1);
	setOutputChannels(getAmountFilters());
	setImgSizeOneChannel(std::pow(getInputDimX(),2));
	setFilterSizeOneChannel(std::pow(getInputDimW(),2));
	setOutputSizeOneFilter(std::pow(getOutputDim(),2));
	setOutputSize( getOutputSizePerChannel() * getOutputChannels());
	/*
	 *
	 */

	//TODO for efficency maybe change eigen to row major order for convfilter this should make sense, not sure about the
	// other operations ( as long as each input is a row)

	double convolutionCounter=0;

	Eigen::MatrixXf outputMatrix = Eigen::MatrixXf::Zero(getAmountOfInputs(), getOutputSize());
	//loop over all images :

    for (int i = 0; i < getAmountOfInputs(); i++) {
	    //DO convolution for image i
		for(int c =0;c< getInputChannels();c++){
		    //get the channel at c
            Eigen::MatrixXf currentChannel = getInputA()->getForward().block(i,_imgSizeOneChannel*c,1,_imgSizeOneChannel);
            currentChannel.resize(getInputDimX(),getInputDimX());

            //2. loop over all filters
            for (int j = 0; j < getAmountFilters(); j++) {
                //get the kernel at j
                Eigen::MatrixXf currentKernel = getInputB()->getForward().block(j,_filterSizeOneChannel*c,1,_filterSizeOneChannel);
                currentKernel.resize(getInputDimW(),getInputDimW());
                convolutionCounter++;
                auto convolvedOutput = convolve(currentChannel,currentKernel,_stride,getOutputDim());

                //Transpose is necessary because of resize operation
                convolvedOutput.transposeInPlace();
                convolvedOutput.resize(1,_outputSizeOneFilter);
                outputMatrix.block(i,_outputSizeOneFilter*j,1,_outputSizeOneFilter)+= convolvedOutput.block(0,0,1,_outputSizeOneFilter);

            }

        }

	}

	setForward(outputMatrix);


    stopTimeMeasurement(0);
//    std::cout<<"Amount of forward convolutions:"<<convolutionCounter<<std::endl;

};

void ConvolveFilter::backwards() {
    startTimeMeasurement();
    double convolutionCounter=0;

//Gradient of Kernel is convolution of inputA with the gradient
    Eigen::MatrixXf gradientsKernel = Eigen::MatrixXf::Zero(getAmountFilters(), _filterSizeOneChannel * getInputChannels());
    Eigen::MatrixXf gradientsInput = Eigen::MatrixXf::Zero(getAmountOfInputs(), getImgSizeOneChannel() * getInputChannels());

#pragma parallel for num_threads(8)

    //loop over all images :
	for (int i = 0; i < getAmountOfInputs(); i++) {
	    //DO convolution for image i

             for (int j = 0; j < getAmountFilters(); j++) {
                 Eigen::MatrixXf currentGradients = getCurrentGradients().block(i,_outputSizeOneFilter*j,1,_outputSizeOneFilter);//.block(j,_filterSizeOneChannel*c,1,_filterSizeOneChannel);
                 currentGradients.resize(getOutputDim(),getOutputDim());
                 addStrideDilation(currentGradients,_stride);

                 Eigen::MatrixXf currentGradientPadded =currentGradients;
                 //APPLY pADDING
                 addPadding(currentGradientPadded,getInputDimW()-1,getInputDimW()-1);


                 for(int c =0;c< getInputChannels();c++){
                    //get the channel at c
                    Eigen::MatrixXf currentInputA = getInputA()->getForward().block(i,_imgSizeOneChannel*c,1,_imgSizeOneChannel);
                     currentInputA.resize(getInputDimX(),getInputDimX());

                     Eigen::MatrixXf currentInputB = getInputB()->getForward().block(j,_filterSizeOneChannel*c,1,_filterSizeOneChannel);//.block(j,_filterSizeOneChannel*c,1,_filterSizeOneChannel);
                     currentInputB.resize(getInputDimW(),getInputDimW());
                     currentInputB.reverse().eval();
                     auto convolvedX = convolve(currentGradientPadded,currentInputB,1,getInputDimW());
                     convolutionCounter++;


                     auto convolvedW = convolve(currentInputA,currentGradients,_stride,getInputDimW());
                     convolutionCounter++;


                     //Transpose is necessary because of resize operation
                     convolvedW.transposeInPlace();
                     convolvedW.resize(1,_filterSizeOneChannel);
                    gradientsKernel.block(j,_filterSizeOneChannel*c,1,_filterSizeOneChannel)+=convolvedW.block(0,0,1,_filterSizeOneChannel);

                     convolvedX.transposeInPlace();
                     convolvedX.resize(1,_imgSizeOneChannel);
                     gradientsInput.block(i,_imgSizeOneChannel*c,1,_imgSizeOneChannel)+=convolvedX.block(0,0,1,_imgSizeOneChannel);

            }

        }

	}

	getInputB()->setCurrentGradients(gradientsKernel);
    getInputA()->setCurrentGradients(gradientsInput);

    stopTimeMeasurement(1);
//    std::cout<<"Amount of forward convolutions:"<<convolutionCounter<<std::endl;


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

int ConvolveFilter::getStride() const {
    return _stride;
}

void ConvolveFilter::setStride(int stride) {
    _stride = stride;
}

int ConvolveFilter::getAmountFilters() const {
    return _amountFilters;
}

void ConvolveFilter::setAmountFilters(int amountFilters) {
    _amountFilters = amountFilters;
}

int ConvolveFilter::getImgSizeOneChannel() const {
    return _imgSizeOneChannel;
}

void ConvolveFilter::setImgSizeOneChannel(int imgSizeOneChannel) {
    _imgSizeOneChannel = imgSizeOneChannel;
}

int ConvolveFilter::getFilterSizeOneChannel() const {
    return _filterSizeOneChannel;
}

void ConvolveFilter::setFilterSizeOneChannel(int filterSizeOneChannel) {
    _filterSizeOneChannel = filterSizeOneChannel;
}

int ConvolveFilter::getOutputSizePerChannel() const {
    return _outputSizeOneFilter;
}

void ConvolveFilter::setOutputSizeOneFilter(int outputSizeOneFilter) {
    _outputSizeOneFilter = outputSizeOneFilter;
}

int ConvolveFilter::getOutputSize() const {
    return _outputSize;
}

void ConvolveFilter::setOutputSize(int outputSize) {
    _outputSize = outputSize;
}
