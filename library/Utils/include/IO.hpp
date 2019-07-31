//
// Created by pbo on 04.07.19.
//

#pragma once

#include <commonDatatypes.hpp>
#include <Eigen/Dense>
/*!
 * Writes an Matrix to binary
 * @param filename
 * @param matrix
 * @return
 */
bool write_binary(const std::string& filename, const Matrix &matrix);

/*!
 * reads an binary into a eigen matrix
 * @param filename
 * @param matrix
 * @return
 */
bool read_binary(const std::string& filename, Matrix &matrix);




