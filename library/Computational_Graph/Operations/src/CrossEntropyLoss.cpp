//
// Created by phili on 30.06.2019.
//

#include <iostream>
#include "CrossEntropyLoss.hpp"

void CrossEntropyLoss::forwards() {
    startTimeMeasurement();

    /*
 * GENERALL STUFF
 */
//    setOutputChannels(getInputA()->getOutputChannels());
	beforeForward();/*
 *
 */
//	negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))
/*
	std::cout<<"input:\n"<<getInputA()->getForward()<<std::endl;
*/
    int rowsA = getInputA()->getForward().rows();
    int rowsB = getInputB()->getForward().rows();
    int colsA = getInputA()->getForward().cols();
    int colsB = getInputB()->getForward().cols();

	Eigen::MatrixXf log = Eigen::log(getInputA()->getForward().array());
//	std::cout<<"log:\n"<<log<<std::endl;
	Eigen::MatrixXf multiply = log.cwiseProduct(getInputB()->getForward());
//	std::cout<<"multiply:\n"<<multiply<<std::endl;

	auto sumC = multiply.sum();
//	std::cout<<"sumC:\n"<<sumC<<std::endl;

//	auto sumN = sumC.rowwise().sum();
////	Eigen::MatrixXf minus = Eigen::MatrixXf::Ones(sumN.rows(),sumN.cols());
	float minus = sumC*-1;

	Eigen::MatrixXf result(log.rows(),log.cols());

	for (int i = 0; i < result.rows(); i++) {
		for(int j = 0;j<result.cols();j++)
		result(i, j) = minus;
	}

//	Eigen::MatrixXf result = (-1)*(getInputB()->getForward().dot(log.transpose().eval())).sum();
//result/=log.rows();
	setForward(result);
    stopTimeMeasurement(0);

}


void CrossEntropyLoss::backwards() {
    startTimeMeasurement();

    //– ci / pi + (1 – ci)/ (1 – pi)
	Eigen::MatrixXf c = getInputB()->getForward()  ;
	Eigen::MatrixXf p = getInputA()->getForward() ;
//	Eigen::MatrixXf p2 = getForward() ;
//	Eigen::MatrixXf tmp = c.cwiseQuotient(p);
	Eigen::MatrixXf tmp2 = p-c;
	tmp2=tmp2/getInputA()->getForward().rows();
//	tmp2.array()+=0.0000000000000000000000000000000001;


	getInputA()->setCurrentGradients(tmp2);
    stopTimeMeasurement(1);

}

