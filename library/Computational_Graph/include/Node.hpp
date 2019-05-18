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

	virtual void forwards(){};
	virtual void backwards(float previousGradient){};
	/**
	 * This method is used to identify the type of the instances
	 * Therefore the following numbers apply
	 * Operation :=1
	 * Placeholder :=2
	 */
//	int getType(){return _id;};
//protected:
//	  int _id;
	 float _forwardData;
	 float _backwardData;
	Eigen::VectorXf _forwardCache;

	Eigen::VectorXf _gradients;
virtual const std::vector<std::shared_ptr<Node>> &getInputNodes(){return _inputNodes;} ;

private:
	std::vector<std::shared_ptr<Node>> _inputNodes;
	std::vector<std::shared_ptr<Node>> _outputNodes;



};


