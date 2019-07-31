//
// Created by phili on 15.05.2019.
//
//
// Created by phili on 14.05.2019.
//
#include <MultiplicationOp.hpp>
#include <Placeholder.hpp>
#include <memory>

#include <catch2/catch.hpp>
#include <Parameter.hpp>
#include <iostream>
#include <SigmoidOP.hpp>
#include <MSEOp.hpp>
#include <SummationOp.hpp>
#include <Graph.hpp>
#include <OperationsFactory.hpp>

TEST_CASE("Multiplication Forwardpass ", "[operation]") {

	SECTION("without activation functions and one input vector", "[XOR Computational Graph]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf mX1(1, 2);
		mX1 << 2, 10;

		Eigen::MatrixXf W1(2, 2);
		W1 << 1, 3,
				2, 4;

		Eigen::MatrixXf W2(2, 1);
		W2 << 6, 7;


		auto X = std::make_shared<Placeholder>(mX1);
		graph->setInput(X);
		auto mul = OperationsFactory::createMultiplicationOp(graph, X, W1);
		auto mul2 = OperationsFactory::createMultiplicationOp(graph, mul, W2);

		graph->computeForward();

		REQUIRE(mul->getForward()(0, 0) == 22);
		REQUIRE(mul->getForward()(0, 1) == 46);
		REQUIRE(mul2->getForward()(0, 0) == 454);


	}

}

TEST_CASE("Multiplication BackwardPass Parameter ", "[operation]") {

	auto graph = std::make_shared<Graph>();

	Eigen::MatrixXf mX1(1, 2);
	mX1 << 2, 10;

	Eigen::MatrixXf W1(2, 2);
	W1 << 1, 3,
			2, 4;

	Eigen::MatrixXf W2(2, 1);
	W2 << 6, 7;


	auto X = std::make_shared<Placeholder>(mX1);
	graph->setInput(X);
	auto mul = OperationsFactory::createMultiplicationOp(graph, X, W1);
	auto mul2 = OperationsFactory::createMultiplicationOp(graph, mul, W2);

	graph->computeForward();
	graph->computeBackwards();
	graph->updateParameters();


	REQUIRE(mul2->getParameter()->getPreviousGradients()(0, 0) == 22);
	REQUIRE(mul2->getParameter()->getPreviousGradients()(1, 0) == 46);

	REQUIRE(mul->getParameter()->getPreviousGradients()(0, 0) == 12);
	REQUIRE(mul->getParameter()->getPreviousGradients()(0, 1) == 14);
	REQUIRE(mul->getParameter()->getPreviousGradients()(1, 0) == 60);
	REQUIRE(mul->getParameter()->getPreviousGradients()(1, 1) == 70);


}

TEST_CASE("Multiplication BackwardPass Input ", "[operation]") {

	auto graph = std::make_shared<Graph>();

	Eigen::MatrixXf mX1(1, 2);
	mX1 << 2, 10;

	Eigen::MatrixXf W1(2, 2);
	W1 << 1, 3,
			2, 4;

	Eigen::MatrixXf W2(2, 1);
	W2 << 6, 7;


	auto X = std::make_shared<Placeholder>(mX1);
	graph->setInput(X);
	auto mul = OperationsFactory::createMultiplicationOp(graph, X, W1);
	auto mul2 = OperationsFactory::createMultiplicationOp(graph, mul, W2);

	graph->computeForward();
	graph->computeBackwards();

	REQUIRE(X->getPreviousGradients()(0, 0) == 27);
	REQUIRE(X->getPreviousGradients()(0, 1) == 40);


}

