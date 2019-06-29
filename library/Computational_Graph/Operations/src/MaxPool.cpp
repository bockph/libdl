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
    Eigen::MatrixXf indexMatrix = Eigen::MatrixXf::Zero(imgN, imgDim*imgDim);

    //loop over all images :
    for (int i = 0; i < imgN; i++) {

        for (int c = 0; c < getInputA()->getOutputChannels(); c++) {
            Eigen::MatrixXf tmpIMG = getInputA()->getForward().block(i, imgSizeOneChannel * c, 1, imgSizeOneChannel);
            tmpIMG.resize(imgDim, imgDim);
            tmpIMG.transposeInPlace();
            Eigen::MatrixXf indexTMP=Eigen::MatrixXf::Zero(imgDim,imgDim);




            //2.2 loop over amount of possible convolutions and apply filter to img
            for (int x = 0; x < outputDim; x++) {
                for (int y = 0; y < outputDim; y++) {
                    int x_stride = x * _stride;
                    int y_stride = y * _stride;
                    outputMatrix(i, y + x * outputDim + c * outputDim * outputDim) =
                            tmpIMG.block(x_stride, y_stride, 2, 2).maxCoeff(&maxRow, &maxCol);
                    indexTMP.block(x_stride, y_stride, 2, 2)(i, maxCol + maxRow * outputDim + c * outputDim * outputDim) = 1;

                }
            }
            indexTMP.transposeInPlace();
            indexTMP.resize(1,imgDim*imgDim);
            indexMatrix.block(i,imgDim*imgDim*c,1,imgDim*imgDim)= indexTMP.block(0,0,1,imgDim*imgDim);

        }



    }
    setMaxIndexMatrix(indexMatrix);
    setForward(outputMatrix);


};

void MaxPool::backwards() {
    int imgN = getInputA()->getForward().rows();
    int imgDim = getInputA()->getOutputDim();
    int outputDim = std::floor((imgDim - _windowSize) / _stride) + 1;
    int imgSizeOneChannel = std::pow(imgDim, 2);


    Eigen::MatrixXf::Index maxRow, maxCol;


    Eigen::MatrixXf indexMatrix = Eigen::MatrixXf::Zero(imgN, imgDim*imgDim);


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
                    tmpIMG.block(x_stride, y_stride, 2, 2)*=getCurrentGradients()(i, y + x * outputDim + c * outputDim * outputDim);

                }
            }

            tmpIMG.transposeInPlace();
            tmpIMG.resize(1,imgDim*imgDim);
            indexMatrix.block(i,imgDim*imgDim*c,1,imgDim*imgDim)= tmpIMG.block(0,0,1,imgDim*imgDim);

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
