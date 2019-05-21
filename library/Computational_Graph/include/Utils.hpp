//
// Created by phili on 21.05.2019.
//

#include <Eigen/Dense>

Eigen::MatrixXf generateMatrix(float LO,float HI,int rows,int cols){

	float range= HI-LO;
	Eigen::MatrixXf m = Eigen::MatrixXf::Random(rows,cols); // 3x3 Matrix filled with random numbers between (-1,1)
	m = (m + Eigen::MatrixXf::Constant(rows,cols,1.))*range/2.; // add 1 to the matrix to have values between 0 and 2;
	// multiply with range/2
	m = (m + Eigen::MatrixXf::Constant(rows,cols,LO)); //set LO as the lower bound (offset)
	return m;
}
