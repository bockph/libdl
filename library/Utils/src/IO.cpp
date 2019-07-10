//
// Created by pbo on 04.07.19.
//

#include "IO.hpp"
#include <fstream>

    void write_binary(const char* filename, const Eigen::MatrixXf& matrix){
        std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        typename Eigen::MatrixXf::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Eigen::MatrixXf::Index));
        out.write((char*) (&cols), sizeof(typename Eigen::MatrixXf::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Eigen::MatrixXf::Scalar) );
        out.close();
    }
    void read_binary(const char* filename, Eigen::MatrixXf& matrix){
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        typename Eigen::MatrixXf::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Eigen::MatrixXf::Index));
        in.read((char*) (&cols),sizeof(typename Eigen::MatrixXf::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename Eigen::MatrixXf::Scalar) );
        in.close();
    }

