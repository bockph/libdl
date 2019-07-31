//
// Created by pbo on 04.07.19.
//

#include "DataInitialization.hpp"



Eigen::MatrixXf DataInitialization::generateRandomMatrix(int rows, int cols){

    Eigen::MatrixXf m = Eigen::MatrixXf::Random(rows,cols);

    float test =std::sqrt(2/((float)rows-1));

    m = m *test;



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
    //TODO maybe move to NeuralNetwork CleanUp
    float test =std::sqrt(2/((float)rows-1));

    tmp = tmp *test;

    return tmp;
}

