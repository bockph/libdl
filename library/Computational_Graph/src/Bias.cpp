//
// Created by phili on 21.05.2019.
//


#include <iostream>
#include "Bias.hpp"

Bias::Bias(Eigen::MatrixXf& m, int channel) {
	setForward(m);
	setOutputChannels(channel);
    _v1 = Eigen::MatrixXf::Zero(m.rows(),m.cols());
    _s1 = Eigen::MatrixXf::Zero(m.rows(),m.cols());
}


void Bias::backwards() {
	int rowsG = getCurrentGradients().rows();
	int rowsCurrent = getForward().rows();
	bool adam(false);
	//TODO: Implement Learning Rate
	if (rowsCurrent != rowsG) {
		if(adam){
			_v1 =_v1.replicate(rowsG,1);
			_s1 = _s1.replicate(rowsG,1);
			Eigen::MatrixXf tmp = Eigen::MatrixXf::Zero(getCurrentGradients().rows(),getCurrentGradients().cols());
			tmp =(getCurrentGradients()/BATCH_SIZE);
			tmp = tmp.array().pow(2);
			tmp *=(1-beta2);
			_v1 = beta1*_v1 + (1-beta1)/BATCH_SIZE*getCurrentGradients();// # momentum update
			_s1 = beta2*_s1 +tmp;//# RMSProp update
			Eigen::MatrixXf f1 = getForward();
			Eigen::MatrixXf div = (_s1.array()+1e-7).array().sqrt();
			f1 =f1.replicate(rowsG,1)- lr * _v1.cwiseQuotient(div);

			setForward(f1);
		}else{
			setForward(getForward().replicate(rowsG, 1) - lr* getCurrentGradients());

		}

	} else {
		if(adam){
			Eigen::MatrixXf tmp = Eigen::MatrixXf::Zero(getCurrentGradients().rows(),getCurrentGradients().cols());
			tmp =(getCurrentGradients()/BATCH_SIZE);
			tmp = tmp.array().pow(2);
			tmp *=(1-beta2);
			_v1 = beta1*_v1 + (1-beta1)/BATCH_SIZE*getCurrentGradients();// # momentum update
			_s1 = beta2*_s1 +tmp;//# RMSProp update
			Eigen::MatrixXf f1 = getForward();
			Eigen::MatrixXf div = (_s1.array()+1e-7).array().sqrt();
			f1 -= lr * _v1.cwiseQuotient(div);

			setForward(f1);
		}else{
			setForward(getForward() - lr * getCurrentGradients());
		}

	}

}