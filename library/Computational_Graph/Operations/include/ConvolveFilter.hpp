//
// Created by phili on 14.06.2019.
//

#pragma once


#include <Operation.hpp>
#include <Variable.hpp>
#include <Placeholder.hpp>

class ConvolveFilter : public Operation {
public:
	ConvolveFilter(std::shared_ptr<Node> X, std::shared_ptr<Filter> W,int stride =1);

	~ConvolveFilter() = default;


    void addPadding(Eigen::MatrixXf& m, int rowPadding, int colPadding);
    void addStrideDilation(Eigen::MatrixXf &m, int stride);

    int getStride() const;

    void setStride(int stride);

    int getAmountFilters() const;

    void setAmountFilters(int amountFilters);

    Eigen::MatrixXf convolve(const Eigen::MatrixXf& input, const Eigen::MatrixXf& filter,int stride,int outputDim);
	void forwards() override;

	void backwards() override;


    int getImgSizeOneChannel() const;

    void setImgSizeOneChannel(int imgSizeOneChannel);

    int getFilterSizeOneChannel() const;

    void setFilterSizeOneChannel(int filterSizeOneChannel);

    int getOutputSizePerChannel() const;

    void setOutputSizeOneFilter(int outputSizeOneFilter);

    int getOutputSize() const;

    void setOutputSize(int outputSize);

private:
	int _stride;
	int _amountFilters;
    int _imgSizeOneChannel;
    int _filterSizeOneChannel;
    int _outputSizeOneFilter;
    int _outputSize;

};
