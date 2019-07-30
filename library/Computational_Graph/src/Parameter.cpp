//
// Created by phili on 14.06.2019.
//

#include <iostream>
#include "Parameter.hpp"


Parameter::Parameter(Matrix& filter, int channel) {
	setForward(filter);
	setOutputChannels(channel);
	_v1 = Matrix::Zero(filter.rows(),filter.cols());
	_s1 = Matrix::Zero(filter.rows(),filter.cols());
}
Parameter::Parameter(Matrix& matrix) {
	setForward(matrix);
	_v1 = Matrix::Zero(matrix.rows(),matrix.cols());
	_s1 = Matrix::Zero(matrix.rows(),matrix.cols());
}

void Parameter::updateVariable(const HyperParameters& params) {
	Matrix tmp = getPreviousGradients()/params._batchsize;

    switch(params._optimizer){
        case Optimizer::Adam:
        {
            Matrix tmpPow =tmp.array().pow(2);
            _v1 = params._beta1*_v1 + (1-params._beta1)*tmp;// # momentum update
            _s1 = params._beta2*_s1 + (1-params._beta2)*tmpPow;//# RMSProp update
            Eigen::MatrixXf f1 = getForward();
            Eigen::MatrixXf div = (_s1.array()+1e-7).array().sqrt();
            f1 -= params._learningRate * _v1.cwiseQuotient(div);

            setForward(f1);
            break;
        }
        default:
            setForward(getForward()-params._learningRate*tmp);
            break;
    }




}




