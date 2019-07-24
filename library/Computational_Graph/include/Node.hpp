//
// Created by phili on 08.05.2019.
//
#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>



using Matrix = Eigen::MatrixXf;

class Node {
public:
    Node();

    ~Node() = default;


//    virtual void addOutputNode(std::shared_ptr<Node> n);
    virtual const std::vector<std::shared_ptr<Node>> &getInputNodes() { return _inputNodes; };


    const Eigen::MatrixXf &getForward() const;

    void setForward(const Eigen::MatrixXf &forward);

    const std::shared_ptr<Node> &getInputA() const;

    void setInputA(const std::shared_ptr<Node> &inputA);

    const std::shared_ptr<Node> &getInputB() const;

    void setInputB(const std::shared_ptr<Node> &inputB);


    const Eigen::MatrixXf &getCurrentGradients() const;

    void setCurrentGradients(const Eigen::MatrixXf &currentGradients);

    void setOutputChannels(int outputChannels);

    int getOutputChannels() const;

    int getInputChannels() const;


    void setInputChannels(int inputChannels);


    virtual void forwards() {};

    virtual void backwards() {};


    Eigen::MatrixXf _forward;
    Eigen::MatrixXf currentGradients;


private:
    std::vector<std::shared_ptr<Node>> _inputNodes;
    std::shared_ptr<Node> inputA;
    std::shared_ptr<Node> inputB;
    int _outputChannels;
    int _inputChannels;


};


