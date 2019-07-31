//
// Created by pbo on 04.07.19.
//

#pragma once

#include <Eigen/Dense>

bool write_binary(std::string filename, const Eigen::MatrixXf &matrix);

bool read_binary(std::string filename, Eigen::MatrixXf &matrix);




