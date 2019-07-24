//
// Created by phili on 14.06.2019.
//
#pragma once

#include <Node.hpp>
#include <commonDatatypes.hpp>


class Variable : public Node {

public:
    Variable(Eigen::MatrixXf &t, int channel = 1, int dim = 1);

    ~Variable() = default;

    void backwards() override;

    float getLearningRate() const;

    void setLearningRate(float learningRate);

    const hyperParameters &getHyperParameters() const;

    void setHyperParameters(const hyperParameters &hyperParameters);

private:
    Eigen::MatrixXf _s1;
    Eigen::MatrixXf _v1;
    hyperParameters _hyperParameters;

};


