//
// Created by phili on 08.05.2019.
//
#pragma once

#include "Node.hpp"
#include <vector>
#include <memory>


class Operation : public Node {

public:
	Operation(std::shared_ptr<Node> X, std::shared_ptr<Node> W);

	Operation(std::shared_ptr<Node> X);

    int getInputDimX() const;

    void setInputDimX(int inputDimX);

    int getInputDimW() const;

    void setInputDimW(int inputDimW);

    const std::vector<std::shared_ptr<Node>> &getInputNodes() override;
	virtual std::string printForward();
    void beforeForward();

    int getAmountOfInputs() const;

    void setAmountOfInputs(int amountOfInputs);

private:
	std::vector<std::shared_ptr<Node>> _inputNodes;
    int _amountOfInputs;
    int _inputDimX;
    int _inputDimW;

};


