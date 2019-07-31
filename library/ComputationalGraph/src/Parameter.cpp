//
// Created by phili on 14.06.2019.
//

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

void Parameter::updateParameter(const HyperParameters &hyperParameters) {
	Matrix tmp = getPreviousGradients()/hyperParameters._batchSize;

    switch(hyperParameters._optimizer){
        case Optimizer::Adam:
        {
            Matrix tmpPow =tmp.array().pow(2);
            _v1 = hyperParameters._beta1*_v1 + (1-hyperParameters._beta1)*tmp;// # momentum update
            _s1 = hyperParameters._beta2*_s1 + (1-hyperParameters._beta2)*tmpPow;//# RMSProp update
            Eigen::MatrixXf f1 = getForward();
            Eigen::MatrixXf div = (_s1.array()+1e-7).array().sqrt();
            f1 -= hyperParameters._learningRate * _v1.cwiseQuotient(div);

            setForward(f1);
            break;
        }
        default:
            setForward(getForward()-hyperParameters._learningRate*tmp);
            break;
    }




}




