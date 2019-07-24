//
// Created by phili on 14.06.2019.
//

#include <iostream>
#include "Variable.hpp"


Variable::Variable(Eigen::MatrixXf& m, int channel,int dim) {
	setForward(m);
//    setOutputDim(dim);
    setOutputChannels(channel);
    _v1 = Eigen::MatrixXf::Zero(m.rows(),m.cols());
    _s1 = Eigen::MatrixXf::Zero(m.rows(),m.cols());
    /*
     * A Variable should be a row vector (for the beginning ) of the Form RGB, RGB,RGB,... so 3 data points per pixel
     * If several filters are used, a matrix -existing of several row vectors- is used
     */

}

void Variable::backwards() {
    Matrix tmp =getCurrentGradients()/getHyperParameters()._batchsize;

    switch(_hyperParameters._optimizer){
        case Optimizer::Adam:
        {
            Matrix tmpPow =tmp.array().pow(2);
            _v1 = getHyperParameters()._beta1*_v1 + (1-getHyperParameters()._beta1)*tmp;// # momentum update
            _s1 = getHyperParameters()._beta2*_s1 + (1-getHyperParameters()._beta2)*tmpPow;//# RMSProp update
            Eigen::MatrixXf f1 = getForward();
            Eigen::MatrixXf div = (_s1.array()+1e-7).array().sqrt();
            f1 -= getHyperParameters()._learningRate * _v1.cwiseQuotient(div);

            setForward(f1);
            break;
        }
        default:
            setForward(getForward()-getHyperParameters()._learningRate*tmp);
            break;
    }




}



const hyperParameters &Variable::getHyperParameters() const {
    return _hyperParameters;
}

void Variable::setHyperParameters(const hyperParameters &hyperParameters) {
    _hyperParameters = hyperParameters;
}
