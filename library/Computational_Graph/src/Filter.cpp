//
// Created by phili on 14.06.2019.
//

#include <iostream>
#include "Filter.hpp"


Filter::Filter(Eigen::MatrixXf m,int dim, int channel) {
	setForward(m);
    setOutputDim(dim);
    setOutputChannels(channel);
    _v1 = Eigen::MatrixXf::Zero(m.rows(),m.cols());
    _s1 = Eigen::MatrixXf::Zero(m.rows(),m.cols());
    /*
     * A Filter should be a row vector (for the beginning ) of the Form RGB, RGB,RGB,... so 3 data points per pixel
     * If several filters are used, a matrix -existing of several row vectors- is used
     */

}

void Filter::backwards() {
    Eigen::MatrixXf tmp = Eigen::MatrixXf::Zero(getCurrentGradients().rows(),getCurrentGradients().cols());
    tmp =(getCurrentGradients()/BATCH_SIZE); // ;
    tmp = tmp.array().pow(2);
    tmp *=(1-beta2);
    _v1 = beta1*_v1 + (1-beta1)/BATCH_SIZE*getCurrentGradients();// # momentum update
    _s1 = beta2*_s1 +tmp;//# RMSProp update
    Eigen::MatrixXf f1 = getForward();
    Eigen::MatrixXf div = (_s1.array()+1e-7).array().sqrt();
    f1 -= alpha * _v1.cwiseQuotient(div);

    setForward(f1);
    /*std::cout<<"\nFilter:\n"<<getForward();
    std::cout<<"\n Gradients Filter:\n"<<getCurrentGradients()<<std::endl;*/
//	setForward(getForward() - 0.001 * getCurrentGradients());
}