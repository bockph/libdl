//
// Created by phili on 17.05.2019.
//

#include <iostream>
#include "Weight.hpp"


Weight::Weight(Eigen::MatrixXf m) {
	setForward(m);
    _v1 = Eigen::MatrixXf::Zero(m.rows(),m.cols());
    _s1 = Eigen::MatrixXf::Zero(m.rows(),m.cols());
}

void Weight::backwards() {

	bool adam(false);
	if(adam){
		Eigen::MatrixXf tmp = Eigen::MatrixXf::Zero(getCurrentGradients().rows(),getCurrentGradients().cols());
		tmp =(getCurrentGradients() /BATCH_SIZE);
//    std::cout<<"SSSSSS;"<<std::endl;

//    std::cout<<tmp<<std::endl;
		tmp = tmp.array().pow(2);
		tmp *=(1-beta2);
		_v1 = beta1*_v1 + (1-beta1)/BATCH_SIZE*getCurrentGradients();// # momentum update
//    std::cout<<"VVVV;"<<std::endl;

//    std::cout<<_v1<<std::endl;
		_s1 = beta2*_s1 +tmp;//# RMSProp update
		Eigen::MatrixXf f1 = getForward();
		Eigen::MatrixXf div = (_s1.array()+1e-7).array().sqrt();
		f1 -= lr * _v1.cwiseQuotient(div);

		setForward(f1);
	} else{
		setForward(getForward() - lr * getCurrentGradients());
	}

}