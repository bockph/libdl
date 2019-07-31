//
// Created by pbo on 04.07.19.
//


#pragma once
#include <Eigen/Dense>
#include <random>
class DataInitialization {
public:


static Eigen::MatrixXf generateRandomMatrix(float LO, float HI, int rows, int cols);
static Eigen::MatrixXf generateRandomMatrixNormalDistribution(float mean, float stdev, int rows, int cols);

};


