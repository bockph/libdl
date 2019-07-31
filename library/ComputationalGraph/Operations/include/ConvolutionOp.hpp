//
// Created by phili on 14.06.2019.
//

#pragma once


#include <Parameter.hpp>
#include "NormalFunction.hpp"

/*!
 * Implements a Convolution Operation
 */
class ConvolutionOp : public NormalFunction {
public:
    /*!
     * - creates a Normal Function and sets the outputChannel to the amount of Filters passed
     * - checks whether X and filter do have the same amount of channels if not a exception is thrown
     *
     * @param X input Noode
     * @param filter all kernels that should be applied. Each row of the forward pass represents one kernel
     * @param stride stride
     */
    ConvolutionOp(std::shared_ptr<Node> X, std::shared_ptr<Parameter> filter, int stride = 1);

    ~ConvolutionOp() override = default;//! default destructor
    /*!
     * - executes a convolution of the input Nodes X and filter
     */
    void forwardPass() override;

    /*!
     * - calculates the gradients of the convolution using getPreviousGradients() and the previous calculated im2Col Matrix
     */
    void backwardPass() override;

    /*!
     * /
	 * - calculates the dimensions of each sample of the minibatch and each kernel of the filter with the amount of channels
     * - transforms the miniBatch into the im2Col format and stores it in im2Col
	 * - executes a convolution by multiplying the filter with im2Col and stores it in outputMatrix
     *
     * @param miniBatch reference to miniBatch matrix where each row corresponds to one sample, channels are stored consecutively , miniBatch.row(i)=sample = [channel 0]...[channel 1]
     * @param filter rference to miniBatch matrix where each row corresponds to one kernel, channels are stored consecutively filter.row(i) = kernel = [channel 0]...[channel 1]
     * @param im2Col reference to Matrix which should store the im2Col output
     * @param outputMatrix reference to Matrix which should store the convolution output of miniBatch with filter
     * @param stride stride
     * @param channels input channels of filter /miniBatch
     */
    static void forwardsConvolution(const Matrix &miniBatch, const Matrix &filter, Matrix &im2Col, Matrix &outputMatrix,
                                    const int stride, const int channels);

    /*!
     * - calculates the gradients w.r.t X and filter
     * - dX = filter.transpose() * dout;
     * - dFilter = dout * im2ColM.transpose();
     *
     * @param dX reference to Matrix which should store  the gradients w.r.t X
     * @param dFilter reference to Matrix which should store  the gradients w.r.t filter
     * @param filter the original filter
     * @param dout reference to Matrix which holds the previously calculated forwardpass
     * @param im2ColM reference to Matrix which holds the previously calculated im2Col Matrix
     * @param batchSize batchSize of X
     * @param inputDimX the dimensions of one sample of the input Node X
     * @param stride original stride
     * @param channels the original channels of X and filter
     */
    static void
    backwardsConvolution(Matrix &dX, Matrix &dFilter, const Matrix &filter, const Matrix &dout, const Matrix &im2ColM,
                         int batchSize, int inputDimX, int stride, int channels);

    /*!
     * Implements the im2Col function
     * see also https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html
     *
     * @param output reference to Matrix which should store the calculated im2Col matrix
     * @param input the matrix that should be transformed
     * @param kernelSize
     * @param stride
     * @param channel
     * @param batchSize
     */
    static void
    im2col(Matrix &output, const Matrix &input, int kernelSize, int stride, int channel, int batchSize);

    /*!
     * Implements the col2Im function
     * see also https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html
     *
     * @param output reference to Matrix which should store the calculated col2Im matrix
     * @param input a im2Col matrix
     * @param kernelSize
     * @param origDim the original Dimensions of a sample of the im2Col inputMatrix
     * @param stride
     * @param channel
     * @param batchSize
     */
    static void
    col2im(Matrix &output, const Matrix &input, int kernelSize, int origDim, int stride, int channel, int batchSize);




private:
    int _stride;//!stores the stride for this convolution operation
    Matrix _im2Col; //!caches the calculated _im2Col matrix

    /*
     * Getters & Setters
     */
public:
    const Matrix &getIm2Col() const;

    void setIm2Col(const Matrix &im2Col);

};
