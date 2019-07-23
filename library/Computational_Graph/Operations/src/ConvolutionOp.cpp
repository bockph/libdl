//
// Created by phili on 14.06.2019.
//

#include <iostream>
#include "ConvolveFilterIM2COL.hpp"



//THis need to be moved to the forward class
ConvolveFilterIM2COL::ConvolveFilterIM2COL(std::shared_ptr<Node> X, std::shared_ptr<Filter> W, int stride)
        :
        Operation(X, W), _stride(stride) {
    if(X->getOutputChannels()!= W->getOutputChannels()){
        throw std::invalid_argument("Input X and W should have the same amount of Channels");
    }
    setOutputChannels(W->getForward().rows());

}




void ConvolveFilterIM2COL::im2col(Matrix &output, const Eigen::MatrixXf &input, int filterSize, int stride, int channel,
                             int batchSize) {

    int filterDim = std::sqrt(filterSize);

    int inputSize = input.cols() / channel;
    int inputDim = std::sqrt(inputSize);
    int outputDim = std::floor((inputDim - filterDim) / stride) + 1;
    int outputSize = std::pow(outputDim, 2);

    output = Matrix(filterSize * channel, outputSize * batchSize);

    for (int c = 0; c < channel; c++) {
        int collumn = 0;
        for (int s = 0; s < batchSize; s++) {
            Eigen::MatrixXf tmp = input.block(s, inputSize * c, 1, inputSize);
            tmp.resize(inputDim, inputDim);

            for (int i = 0; i + filterDim <= inputDim; i += stride) {
                for (int j = 0; j + filterDim <= inputDim; j += stride) {
                    Eigen::MatrixXf col = tmp.block(j, i, filterDim, filterDim);
                    col.resize(filterSize, 1);
                    output.block(c * filterSize, collumn, filterSize,
                                           1) = col; //s*outputSize+i / stride + j / stride * outputDim
                    collumn++;
                }
            }
        }
    }

}

void ConvolveFilterIM2COL::col2im(Matrix& output,const Matrix &input, int filterSize, int outputDim, int stride, int channel,
                             int batchSize) {
    int filterDim = std::sqrt(filterSize);
    int outputSize = std::pow(outputDim, 2);

    output= Matrix(batchSize, outputSize * channel);

    for (int c = 0; c < channel; c++) {
        int collumn = 0;
        for (int s = 0; s < batchSize; s++) {

            Eigen::MatrixXf tmp = Eigen::MatrixXf::Zero(outputDim, outputDim);
            for (int i = 0; i + filterDim <= outputDim; i += stride) {
                for (int j = 0; j + filterDim <= outputDim; j += stride) {
                    Eigen::MatrixXf col = input.block(c * filterSize, collumn, filterSize,
                                                      1);//s*outputSize+i / stride + j / stride * inputDim, filterSize, 1);
                    col.resize(filterDim, filterDim);
                    tmp.block(i, j, filterDim, filterDim) += col;
                    collumn++;
                }
            }
            tmp.resize(1, outputSize);
            output.block(s, c * outputSize, 1, outputSize) = tmp;
        }
    }

}

void ConvolveFilterIM2COL::forwardsConvolution(const Matrix &miniBatch, const Matrix &filter, Matrix &im2ColM,
                                               Matrix &outputMatrix, const int stride, const int channels) {
    int amountSamples = miniBatch.rows();
    int sampleSizeOneChannel = miniBatch.cols() / channels;
    int sampleDim = std::sqrt(sampleSizeOneChannel);

    int amountFilters = filter.rows(); //equals outputChannels
    int kernelSize = filter.cols();
    int kernelSizeOneChannel = kernelSize / channels;
    int kernelDim = std::sqrt(kernelSizeOneChannel);

    int outputDim = std::floor(sampleDim - kernelDim) / stride + 1;
    int outputSize = std::pow(outputDim, 2);
    auto start = std::chrono::system_clock::now();

    im2col(im2ColM, miniBatch, kernelSizeOneChannel, stride, channels, amountSamples);
    auto im2t = std::chrono::system_clock::now();

    Eigen::MatrixXf conv = filter * im2ColM;
    auto mult = std::chrono::system_clock::now();

//Reshape extract method
    //TODO change outputForm
    outputMatrix = Eigen::MatrixXf::Zero(amountSamples, outputSize * amountFilters);

    for (int i = 0; i < amountSamples; i++) {
        for (int f = 0; f < amountFilters; f++) {
            outputMatrix.block(i, outputSize * f, 1, outputSize) = conv.block(f, i * outputSize, 1, outputSize);
        }
    }

    auto end2 = std::chrono::system_clock::now();

    int im2 = std::chrono::duration_cast<std::chrono::microseconds>
            (im2t-start).count();
    int mul = std::chrono::duration_cast<std::chrono::microseconds>
            (mult-im2t).count();
    int reshape = std::chrono::duration_cast<std::chrono::microseconds>
            (end2-mult).count();
    int total = std::chrono::duration_cast<std::chrono::microseconds>
            (end2-start).count();
//    std::cout<<"im2:"<<im2<<"mul:"<<mul<<"reshape:"<<reshape<<"total:"<<total<<std::endl;
//    std::cout<<"Percentage im2:"<<im2/(float)total<<"Percentage mul:"<<mul/(float)total<<"Percentage reshape:"<<reshape/(float)total<<std::endl;
}


void ConvolveFilterIM2COL::forwards() {

    Matrix im2ColM, outputMatrix;

    startTimeMeasurement();

    forwardsConvolution(getInputA()->getForward(), getInputB()->getForward(), im2ColM, outputMatrix, _stride,
                        getInputChannels());


    stopTimeMeasurement(0);
    setIm2Col(im2ColM);

    setForward(outputMatrix);


}

void
ConvolveFilterIM2COL::backwardsConvolution(Matrix &dX, Matrix &dW, const Eigen::MatrixXf &filter, const Matrix &dout,
                                           const Matrix &im2ColM, int batchSize, int inputDimX, int stride,
                                           int channelsX) {

    int amountFilter = filter.rows();
    int outputSize = dout.cols() / amountFilter;
    int filterSizeOneChannel = filter.cols() / channelsX;


    Matrix doutReshaped = Matrix::Zero(amountFilter, batchSize * outputSize);
    for (int i = 0; i < batchSize; i++) {
        for (int f = 0; f < amountFilter; f++) {
            doutReshaped.block(f, i * outputSize, 1, outputSize) = dout.block(i, outputSize * f, 1, outputSize);
        }
    }

    Matrix dXBeforeReshape = filter.transpose() * doutReshaped;

    ConvolveFilterIM2COL::col2im(dX,dXBeforeReshape, filterSizeOneChannel, inputDimX, stride, channelsX, batchSize);

    dW = doutReshaped * im2ColM.transpose();

}

void ConvolveFilterIM2COL::backwards() {


    int batchSize = getInputA()->getForward().rows();
    int inputDimX = std::sqrt(getInputA()->getForward().cols() / getInputChannels());

    Matrix gradientX, gradientW;
    startTimeMeasurement();

    backwardsConvolution(gradientX, gradientW, getInputB()->getForward(), getCurrentGradients(), getIm2Col(), batchSize,
                         inputDimX, _stride, getInputChannels());
    //reshape gradients
    stopTimeMeasurement(1);


    getInputB()->setCurrentGradients(gradientW);
    getInputA()->setCurrentGradients(gradientX);


    /*
     * Debug Information
     */
   /* std::cout<<"Convolution FOrward:"<<getForward()<<std::endl;
    std::cout<<"Convolution Backwards:"<<getCurrentGradients()<<std::endl;*/



}



const Eigen::MatrixXf &ConvolveFilterIM2COL::getIm2Col() const {
    return _im2Col;
}

void ConvolveFilterIM2COL::setIm2Col(const Eigen::MatrixXf &im2Col) {
    _im2Col = im2Col;
}
