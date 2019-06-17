//
// Created by pbo on 17.06.19.
//

#include "MaxPool.hpp"

int getChannels(){
    return 1;
}
void MaxPool::forwards() {
    //this results in a Vector containing in each row the result for a different input of the Batch
    //	setForward(getInputA()->getForward() * getInputB()->getForward());
    //TODO for efficency maybe change eigen to row major order for convfilter this should make sense, not sure about the
    // other operations ( as long as each input is a row)
    int imgN = getInputA()->getForward().rows();
    int imgDim = getInputA()->getForward().cols();
    int imgDimSQRT=std::sqrt(imgDim);

    int amountPixels = imgDim/getChannels();
    int amountRows = std::sqrt(amountPixels);
    int amountCols = amountRows*getChannels();


    int outputDimSQRT = std::floor((imgDimSQRT - _windowSize) / _stride) + 1;
    int outputDim = std::pow(outputDimSQRT,2);
    Eigen::MatrixXf outputMatrix = Eigen::MatrixXf::Zero(imgN, outputDim);
    //loop over all images :
    for (int i = 0; i < imgN; i++) {
        //get the Image as Matrix, currently only with greyscale/oneChannel Images
        //TODO: Implement multi channels
        Eigen::MatrixXf tmpIMG = getInputA()->getForward().block(i,0,1,imgDim);
        tmpIMG.resize(imgDimSQRT,imgDimSQRT);
        tmpIMG.transposeInPlace();


            //2.2 loop over amount of possible convolutions and apply filter to img
            for (int x = 0; x < outputDimSQRT; x++) {
                for(int y =0;y< outputDimSQRT; y++){
                    int x_stride = x*_stride;
                    int y_stride = y*_stride;
                    outputMatrix(i, y+x*outputDimSQRT) =
                            tmpIMG.block(x_stride,y_stride,2,2).maxCoeff();
                }
            }



    }

    setForward(outputMatrix);


};

void MaxPool::backwards() {

//	Eigen::MatrixXf inputGradient = getCurrentGradients() * (getInputB()->getForward().transpose());
//	Eigen::MatrixXf weightGradient = (getInputA()->getForward().transpose()) * getCurrentGradients();
//	getInputA()->setCurrentGradients(inputGradient);
//	getInputB()->setCurrentGradients(weightGradient);

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