//
// Created by phili on 14.06.2019.
//

#pragma once


#include <Operation.hpp>
#include <Filter.hpp>
#include <Placeholder.hpp>

class ConvolveFilterIM2COL : public Operation {
public:
    ConvolveFilterIM2COL(std::shared_ptr<Node> X, std::shared_ptr<Filter> W,int stride =1);

	~ConvolveFilterIM2COL() = default;


    void addPadding(Eigen::MatrixXf& m, int rowPadding, int colPadding);


    int getAmountFilters() const;


	void forwards() override;

	void backwards() override;
	std::string printForward() override;


    int getImgSizeOneChannel() const;

    void setImgSizeOneChannel(int imgSizeOneChannel);


    void setFilterSizeOneChannel(int filterSizeOneChannel);

    int getOutputSizePerChannel() const;

    void setOutputSizeOneFilter(int outputSizeOneFilter);

    int getOutputSize() const;

    void setOutputSize(int outputSize);
    static Eigen::MatrixXf im2col(const Eigen::MatrixXf &input, const Eigen::MatrixXf &filter, int stride, int channel,int batchSize);
    static Eigen::MatrixXf col2im(const Eigen::MatrixXf &input, const Eigen::MatrixXf &filter,int origDim, int stride, int channel,int batchSize);


    const Eigen::MatrixXf &getIm2Col() const;

    void setIm2Col(const Eigen::MatrixXf &im2Col);
    std::vector<Matrix> _im2Cols;

private:
	int _stride;
	int _amountFilters;
    int _imgSizeOneChannel;
    int _filterSizeOneChannel;
    int _outputSizeOneFilter;
    int _outputSize;
    Eigen::MatrixXf _im2Col;

};
