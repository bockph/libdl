//
// Created by phili on 08.05.2019.
//
#pragma once
#include <vector>
#include <memory>
#include <Eigen/Dense>
#define alpha 0.01
#define beta1 0.95
#define beta2 0.99
#define BATCH_SIZE 16
class Node {

public:
	Node();
	~Node() = default;


	virtual void addOutputNode(std::shared_ptr<Node> n);



	const Eigen::MatrixXf &getForward() const;

	void setForward(const Eigen::MatrixXf &forward);

	const std::shared_ptr<Node> &getInputA() const;

	void setInputA(const std::shared_ptr<Node> &inputA);

	const std::shared_ptr<Node> &getInputB() const;

	void setInputB(const std::shared_ptr<Node> &inputB);



	const Eigen::MatrixXf &getCurrentGradients() const;

	void setCurrentGradients(const Eigen::MatrixXf &currentGradients);

    void setOutputChannels(int channels);

    void setOutputDim(int outputDim);

    int getOutputChannels() const;

    int getOutputDim() const;

    int getInputDim() const;

    void setInputDim(int inputDim);

    int getInputChannels() const;

    void setInputChannels(int inputChannels);

    virtual void forwards(){};
	virtual void backwards(){};



	Eigen::MatrixXf _forward;
	float currentGradient;
	Eigen::MatrixXf currentGradients;
virtual const std::vector<std::shared_ptr<Node>> &getInputNodes(){return _inputNodes;} ;
//    virtual std::string printForward();

private:
	std::vector<std::shared_ptr<Node>> _inputNodes;
	std::vector<std::shared_ptr<Node>> _outputNodes;
	std::shared_ptr<Node> inputA;
	std::shared_ptr<Node> inputB;
	int _outputChannels;
	int _inputChannels;
	int _outputDim;
	int _inputDim;




};


