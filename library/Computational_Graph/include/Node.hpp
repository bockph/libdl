//
// Created by phili on 08.05.2019.
//
#pragma once
#include <vector>
#include <memory>
#include <Eigen/Dense>

class Node {

public:
//	Node(const Node& n)= default;
//	Node( Node&& n)= default;


	virtual void addOutputNode(std::shared_ptr<Node> n);
	float getForwardData() const;

	void setForwardData(float forwardData);

	float getBackwardData() const;

	void setBackwardData(float backwardData);

	const Eigen::MatrixXf &getForward() const;

	void setForward(const Eigen::MatrixXf &forward);

	const std::shared_ptr<Node> &getInputA() const;

	void setInputA(const std::shared_ptr<Node> &inputA);

	const std::shared_ptr<Node> &getInputB() const;

	void setInputB(const std::shared_ptr<Node> &inputB);

	float getCurrentGradient() const;

	void setCurrentGradient(float currentGradient);

	const Eigen::MatrixXf &getCurrentGradients() const;

	void setCurrentGradients(const Eigen::MatrixXf &currentGradients);

	virtual void forwards(){};
	virtual void backwards(){};

	 float _forwardData;
	 float _backwardData;
	Eigen::VectorXf _forwardCache;

	Eigen::VectorXf _gradients;

	Eigen::MatrixXf _forward;
	float currentGradient;
	Eigen::MatrixXf currentGradients;
virtual const std::vector<std::shared_ptr<Node>> &getInputNodes(){return _inputNodes;} ;

private:
	std::vector<std::shared_ptr<Node>> _inputNodes;
	std::vector<std::shared_ptr<Node>> _outputNodes;
	std::shared_ptr<Node> inputA;
	std::shared_ptr<Node> inputB;



};


