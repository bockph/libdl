//
// Created by pbo on 17.06.19.
//

#include <iostream>
#include "MaxPoolOp.hpp"

void MaxPoolOp::forwards() {

    startTimeMeasurement();


    //TODO for efficency maybe change eigen to row major order for convfilter this should make sense, not sure about the
    // other operations ( as long as each input is a row)

    int imgN = getInputA()->getForward().rows();
    int imgSize = getInputA()->getForward().cols() / getInputChannels();
    int imgDim = std::sqrt(imgSize);

    int outputDim = std::floor((imgDim - _windowSize) / _stride) + 1;


    Eigen::MatrixXf::Index maxRow, maxCol;


    Eigen::MatrixXf outputMatrix = Eigen::MatrixXf::Zero(imgN, outputDim * outputDim * getOutputChannels());
    Eigen::MatrixXf indexMatrix = Eigen::MatrixXf::Zero(imgN, getInputA()->getForward().cols()); //, imgDim * imgDim);

    //loop over all images :
    for (int i = 0; i < imgN; i++) {

        for (int c = 0; c < getInputA()->getOutputChannels(); c++) {
            Eigen::MatrixXf tmpIMG = getInputA()->getForward().block(i, imgSize * c, 1, imgSize);

            tmpIMG.resize(imgDim, imgDim);
//            tmpIMG.transposeInPlace();
            Eigen::MatrixXf indexTMP = indexMatrix.block(i, imgSize * c, 1, imgSize);
            indexTMP.resize(imgDim, imgDim);
//            indexTMP.transposeInPlace();

            //2.2 loop over amount of possible convolutions and apply filter to img
            for (int x = 0; x < outputDim; x++) {
                for (int y = 0; y < outputDim; y++) {
                    int x_stride = x * _stride;
                    int y_stride = y * _stride;
                    outputMatrix(i, y + x * outputDim + c * outputDim * outputDim) =
                            tmpIMG.transpose().block(x_stride, y_stride, 2, 2).maxCoeff(&maxRow, &maxCol);
                    indexTMP.transpose().block(x_stride, y_stride, 2, 2)
                            (maxRow, maxCol) = 1;

                }
            }
//            indexTMP.transposeInPlace();
            indexTMP.resize(1, imgSize);
            indexMatrix.block(i, imgSize * c, 1, imgSize) = indexTMP.block(0, 0, 1, imgSize);

        }


    }
    setMaxIndexMatrix(indexMatrix);
    setForward(outputMatrix);

    stopTimeMeasurement(0);

};

void MaxPoolOp::backwards() {
    startTimeMeasurement();
    double convolutionCounter = 0;

    int imgN = getInputA()->getForward().rows();
    int imgSize = getInputA()->getForward().cols() / getInputChannels();
    int imgDim = std::sqrt(imgSize);
    int outputDim = std::floor((imgDim - _windowSize) / _stride) + 1;


    Eigen::MatrixXf indexMatrix = Eigen::MatrixXf::Zero(imgN, imgSize * getInputA()->getOutputChannels());


    //loop over all images :
    for (int i = 0; i < imgN; i++) {
        for (int c = 0; c < getInputA()->getOutputChannels(); c++) {
            Eigen::MatrixXf tmpIMG = getMaxIndexMatrix().block(i, imgSize * c, 1, imgSize);
            tmpIMG.resize(imgDim, imgDim);
//            tmpIMG.transposeInPlace();
            convolutionCounter++;
            //2.2 loop over amount of possible convolutions and apply filter to img
            for (int x = 0; x < outputDim; x++) {
                for (int y = 0; y < outputDim; y++) {
                    int x_stride = x * _stride;
                    int y_stride = y * _stride;
                    tmpIMG.transpose().block(x_stride, y_stride, 2, 2) *= getCurrentGradients()(i,
                                                                                    y + x * outputDim +
                                                                                    c * outputDim * outputDim);
                }
            }

//            tmpIMG.transposeInPlace();
            tmpIMG.resize(1, imgDim * imgDim);
            indexMatrix.block(i, imgSize * c, 1, imgSize) = tmpIMG;

        }

    }
    getInputA()->setCurrentGradients(indexMatrix);

    stopTimeMeasurement(1);
    /*
     * Debug Information
     */
    /*std::cout<<" MaxPoolOp FOrward:"<<getForward()<<std::endl;
    std::cout<<" MaxPoolOp Backwards:"<<getCurrentGradients()<<std::endl;*/
}


const Eigen::MatrixXf &MaxPoolOp::getMaxIndexMatrix() const {
    return _maxIndexMatrix;
}

void MaxPoolOp::setMaxIndexMatrix(const Eigen::MatrixXf &maxIndexMatrix) {
    _maxIndexMatrix = maxIndexMatrix;
}
