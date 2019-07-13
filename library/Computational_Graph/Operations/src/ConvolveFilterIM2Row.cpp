//
// Created by phili on 14.06.2019.
//

#include <iostream>
#include "ConvolveFilterIM2COL.hpp"

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
void ConvolveFilter::addStridePadding(Eigen::MatrixXf& m, int stride){
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
Eigen::MatrixXf im2ColConvolution(Eigen::MatrixXf& input, const Eigen::MatrixXf& filter,int stride,int channel){
    int filterSize =filter.cols()/channel;
    int filterDim = std::sqrt(filterSize);
    int inputSize = input.cols()/channel;
    int inputDim = std::sqrt(inputSize);
    int outputDim = std::floor((inputDim-filterDim)/stride)+1;
    int outputSize =std::pow(outputDim,2);
    auto start = std::chrono::system_clock::now();

    Eigen::MatrixXf inputTransformed(filter.cols(),outputSize);
//    Eigen::MatrixXf resizedInput(inputDim*channel,inputDim);
    // no preinitialization because of performance issues
//    Eigen::MatrixXf col (filterDim,filterDim);
//    Eigen::MatrixXf tmp(1,inputSize);
    for(int c = 0; c<channel;c++){
        Eigen::MatrixXf tmp =input.block(0,inputSize*c,1,inputSize);
        tmp.resize(inputDim,inputDim);
//        resizedInput.block(c*inputDim,0,inputDim,inputDim)=tmp;
    for(int i = 0;i+filterDim<=inputDim;i+=stride){
        for(int j =0;j+filterDim<=inputDim;j+=stride){

            Eigen::MatrixXf col = tmp.block(i,j,filterDim,filterDim);
//                std::cout<< "i:"<<i<< "j:"<<j<<"c:"<<c<< "\ncol:\n"<<col<<std::endl;

                col.resize(filterSize,1);
                inputTransformed.block(c*filterSize,i/stride+j/stride*outputDim,filterSize,1)=col;

            }



        }
    }
    Eigen::MatrixXf conv = filter*inputTransformed;
//    return conv;
//    auto result = Eigen::MatrixXf(1,conv.rows()*conv.cols());
//    auto result = Eigen::MatrixXf(1,filter.rows()*inputTransformed.cols());



//    for(int i = 0;i<conv.rows();i++){
//        result.block(0,i*conv.cols(),1,conv.cols()) = conv.block(i,0,1,conv.cols());//conv.row(i);
//    }
    conv.transposeInPlace();
    conv.resize(1,conv.size());
    auto end = std::chrono::system_clock::now();

    int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
            (end-start).count();
//    std::cout<<"result copying:"<<elapsed_seconds<<std::endl;
//    std::cout<<"Conv Before:"<<"rows:"<<conv.rows()<<"\n"<<conv<<std::endl;
    int size = conv.size();
//    conv.eval();

//std::cout<<"Conv:"<<"rows:"<<conv.rows()<<"\n"<<conv<<std::endl;
//    std::cout<<"result:\n"<<result<<std::endl;

    return conv;

}

void ConvolveFilter::forwards() {
//    std::cout<<"im2cols"<<std::endl;


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
//    std::cout<<"GOOOOOOOOOOOOOOOOOOOOOOOOOOO"<<std::endl;
	Eigen::MatrixXf outputMatrix = Eigen::MatrixXf::Zero(getAmountOfInputs(), getOutputSize());
	//loop over all images :
    startTimeMeasurement();

    for (int i = 0; i < getAmountOfInputs(); i++) {
	    Eigen::MatrixXf sample = getInputA()->getForward().row(i);


//	    Eigen::MatrixXf tmp =im2ColConvolution(sample,getInputB()->getForward(),_stride,getInputDimX(),getInputChannels());
//	            int rows = tmp.rows();
//	            int cols = tmp.cols();
//	            int outputt = getOutputSize();
//	            int colI = getAmountOfInputs();
        outputMatrix.block(i,0,1,getOutputSize())=im2ColConvolution(sample,getInputB()->getForward(),_stride,getInputChannels());

        //DO convolution for image i
		/*for(int c =0;c< getInputChannels();c++){
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

        }*/

	}
    stopTimeMeasurement(0);

	setForward(outputMatrix);


    std::cout<<"Amount of forward convolutions:"<<convolutionCounter<<std::endl;

};

void ConvolveFilter::backwards() {
    startTimeMeasurement();
    double convolutionCounter=0;

//Gradient of Kernel is convolution of inputA with the gradient
    Eigen::MatrixXf gradientsKernel = Eigen::MatrixXf::Zero(getAmountFilters(), _filterSizeOneChannel * getInputChannels());
    Eigen::MatrixXf gradientsInput = Eigen::MatrixXf::Zero(getAmountOfInputs(), getImgSizeOneChannel() * getInputChannels());


    //loop over all images :
	for (int i = 0; i < getAmountOfInputs(); i++) {
	    //DO convolution for image i
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
             for (int j = 0; j < getAmountFilters(); j++) {
                 Eigen::MatrixXf currentGradients = getCurrentGradients().block(i,_outputSizeOneFilter*j,1,_outputSizeOneFilter);//.block(j,_filterSizeOneChannel*c,1,_filterSizeOneChannel);
                 currentGradients.resize(getOutputDim(),getOutputDim());
                 addStridePadding(currentGradients,_stride);

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

                    //TODO: the convolution stride should be one and not the original stride?
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
    std::cout<<"Amount of forward convolutions:"<<convolutionCounter<<std::endl;


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
