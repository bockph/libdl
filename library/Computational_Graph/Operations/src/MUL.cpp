//
// Created by phili on 10.05.2019.
//

#include <iostream>
#include "MUL.hpp"

void MUL::forwards() {
    /*
 * GENERALL STUFF
 */
//    setOutputChannels(getInputA()->getOutputChannels());
    beforeForward();/*
 *
 */
    int rowsA = getInputA()->getForward().rows();
    int rowsB = getInputB()->getForward().rows();
    int colsA = getInputA()->getForward().cols();
    int colsB = getInputB()->getForward().cols();
	//this results in a Vector containing in each row the result for a different input of the Batch
	setForward(getInputA()->getForward() * getInputB()->getForward());
//    std::cout<<getInputB()->getForward()<<std::endl;


};

void MUL::backwards() {
//    std::cout<<"X:\n"<< getInputA()->getForward()<<std::endl;
//    std::cout<<"W:\n"<< getInputB()->getForward().transpose().eval()<<std::endl;
//    std::cout<<"getCurrentGradientsMUL:\n"<< getCurrentGradients()<<std::endl;

//    std::cout<<getInputB()->getForward()<<std::endl;


	Eigen::MatrixXf inputGradient = getCurrentGradients() * (getInputB()->getForward().transpose());
//    std::cout<<"Gradient XXX MUL:\n"<< inputGradient<<std::endl;

    Eigen::MatrixXf weightGradient = (getInputA()->getForward().transpose()) * getCurrentGradients();
    inputGradient/getCurrentGradients().rows();
	getInputA()->setCurrentGradients(inputGradient);
    weightGradient/=getCurrentGradients().rows();

    getInputB()->setCurrentGradients(weightGradient);
//    std::cout<<"Gradient WWW MUL:\n"<< weightGradient<<std::endl;

/*
    std::cout<<"Output MUL BACKWARDS:\n"<< inputGradient<<std::endl;
    std::cout<<"Output MUL BACKWARDS Max:\n"<< inputGradient.array().maxCoeff()<<std::endl;
    std::cout<<"Output  MUL BACKWARDSAverage:\n"<< inputGradient.mean()<<std::endl;

    int x =2+3;*/
}
std::string MUL::printForward() {
	return "MUL:0";
}
