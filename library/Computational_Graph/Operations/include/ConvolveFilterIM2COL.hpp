//
// Created by phili on 14.06.2019.
//

#pragma once


#include <Operation.hpp>
#include <Variable.hpp>
#include <Placeholder.hpp>

class ConvolveFilterIM2COL : public Operation {
public:
    ConvolveFilterIM2COL(std::shared_ptr<Node> X, std::shared_ptr<Variable> W,int stride =1);

	~ConvolveFilterIM2COL() = default;
	void forwards() override;

	void backwards() override;



    static void im2col(Matrix& output,const Eigen::MatrixXf &input, int filterSize, int stride, int channel,int batchSize);
    static void col2im(Matrix& output,const Eigen::MatrixXf &input,  int filterSize, int origDim, int stride, int channel,int batchSize);

    static void backwardsConvolution(Matrix& dX,Matrix& dW,const Matrix& filter,const Matrix& dout, const Matrix& im2ColM,int batchSize,int inputDimX, int stride,int channels);
    static void forwardsConvolution(const Matrix& miniBatch,const Matrix& filter, Matrix& im2Col, Matrix& outputMatrix, const int stride, const int channels);

        const Eigen::MatrixXf &getIm2Col() const;

    void setIm2Col(const Eigen::MatrixXf &im2Col);

private:
	int _stride;


    Eigen::MatrixXf _im2Col;

};
