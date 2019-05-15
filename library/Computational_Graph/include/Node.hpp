//
// Created by phili on 08.05.2019.
//
#pragma once
#include <vector>
#include <memory>

class Node {

public:
//	Node(const Node& n)= default;
//	Node( Node&& n)= default;


	virtual void addOutputNode(std::shared_ptr<Node> n);

	virtual float getDatavalue() const;

	virtual void setDatavalue(float datavalue);

	virtual void compute(){};
	/**
	 * This method is used to identify the type of the instances
	 * Therefore the following numbers apply
	 * Operation :=1
	 * Placeholder :=2
	 */
//	int getType(){return _id;};
//protected:
//	  int _id;
	 float _datavalue;
virtual const std::vector<std::shared_ptr<Node>> &getInputNodes(){return _inputNodes;} ;

private:
	std::vector<std::shared_ptr<Node>> _inputNodes;
	std::vector<std::shared_ptr<Node>> _outputNodes;



};


