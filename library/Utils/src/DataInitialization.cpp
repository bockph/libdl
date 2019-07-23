//
// Created by pbo on 04.07.19.
//

#include "DataInitialization.hpp"



Eigen::MatrixXf DataInitialization::generateRandomMatrix(float LO, float HI, int rows, int cols){

    float range= HI-LO;
    Eigen::MatrixXf m = Eigen::MatrixXf::Random(rows,cols); // 3x3 Matrix filled with random numbers between (-1,1)
    /*m = (m + Eigen::MatrixXf::Constant(rows,cols,1.))*range/2.; // add 1 to the matrix to have values between 0 and 2;
    // multiply with range/2
    m = (m + Eigen::MatrixXf::Constant(rows,cols,LO)); //set LO as the lower bound (offset)*/
//	m.normalize();

    float test =std::sqrt(2/((float)rows-1));

    m = m *test;

//	m.normalize();


    return m;
}
Eigen::MatrixXf DataInitialization::generateRandomMatrixNormalDistribution(float mean, float stdev, int rows, int cols){
    static std::random_device __randomDevice;
    static std::mt19937 __randomGen(__randomDevice());
    static std::normal_distribution<float> normalDistribution(mean, stdev);
    Eigen::MatrixXf tmp(rows,cols);
    for(int i = 0;i<rows;i++){
        for(int j = 0;j < cols;j++){
            tmp(i,j) = normalDistribution(__randomGen);
        }
    }
    /*float range= HI-LO;
    Eigen::MatrixXf m = Eigen::MatrixXf::Random(rows,cols); // 3x3 Matrix filled with random numbers between (-1,1)
    m = (m + Eigen::MatrixXf::Constant(rows,cols,1.))*range/2.; // add 1 to the matrix to have values between 0 and 2;
    // multiply with range/2
    m = (m + Eigen::MatrixXf::Constant(rows,cols,LO)); //set LO as the lower bound (offset)
//	m.normalize();
    m = m *std::sqrt(2/(cols-1));*/
    float test =std::sqrt(2/((float)rows-1));

    tmp = tmp *test;

    return tmp;
}

Eigen::MatrixXf DataInitialization::initializeFilter(int rows, int cols, float scale){
    int size = rows*cols;
    float stddev = scale/std::sqrt(size);
    auto m =generateRandomMatrixNormalDistribution(0,stddev,rows,cols);
//	std::cout<<m<<std::endl;

    return m;

}
Eigen::MatrixXf DataInitialization::initializeWeights(int rows, int cols){
    int size = rows*cols;
    float stddev = 1;
    auto m= generateRandomMatrixNormalDistribution(0,stddev,rows,cols)*0.01;
//	std::cout<<m<<std::endl;
    return m;

}