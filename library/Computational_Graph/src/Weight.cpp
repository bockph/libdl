//
// Created by phili on 14.06.2019.
//

#include <iostream>
#include "Weight.hpp"


Weight::Weight(Eigen::MatrixXf& m,int dim, int channel) {
	setForward(m);
//    setOutputDim(dim);
    setOutputChannels(channel);
    _v1 = Eigen::MatrixXf::Zero(m.rows(),m.cols());
    _s1 = Eigen::MatrixXf::Zero(m.rows(),m.cols());
    /*
     * A Weight should be a row vector (for the beginning ) of the Form RGB, RGB,RGB,... so 3 data points per pixel
     * If several filters are used, a matrix -existing of several row vectors- is used
     */

}

void Weight::backwards() {
    Matrix tmp =getCurrentGradients()/BATCH_SIZE;

    bool adam=true;
	if(adam){
        Matrix tmpPow =tmp.array().pow(2);
		_v1 = beta1*_v1 + (1-beta1)*tmp;// # momentum update
		_s1 = beta2*_s1 + (1-beta2)*tmpPow;//# RMSProp update
		Eigen::MatrixXf f1 = getForward();
		Eigen::MatrixXf div = (_s1.array()+1e-7).array().sqrt();
		f1 -= lr * _v1.cwiseQuotient(div);

		setForward(f1);
	}else{
		setForward(getForward() - lr * tmp);
	}



}

float Weight::getLearningRate() const {
    return _learningRate;
}

void Weight::setLearningRate(float learningRate) {
    _learningRate = learningRate;
}
