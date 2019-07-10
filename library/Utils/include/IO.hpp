//
// Created by pbo on 04.07.19.
//

#pragma once
#include <Eigen/Dense>



        void write_binary(const char* filename, const Eigen::MatrixXf& matrix);
        void read_binary(const char* filename, Eigen::MatrixXf& matrix);




