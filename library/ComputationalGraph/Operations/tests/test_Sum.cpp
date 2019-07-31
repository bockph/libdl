//
// Created by phili on 14.05.2019.
//
#include <SummationOp.hpp>
#include <Placeholder.hpp>
#include <memory>

#include <catch2/catch.hpp>

#include <Graph.hpp>
#include <OperationsFactory.hpp>

TEST_CASE("SUM ForwardPass ", "[operation]") {

	SECTION("Normal Addition") {

		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf input(1, 1);
		input <<25;
		Eigen::MatrixXf Add(1, 1);
		Add <<50;

		auto X = std::make_shared<Placeholder>(input,1);
		graph->setInput(X);

		auto sum = OperationsFactory::createSummationOp(graph,X,Add,1);
		graph->computeForward();

		Matrix test(1,1);
		test<<75;
		REQUIRE(sum->getForward().isApprox(test));
	}
	SECTION("Two serial Additions") {

		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf input(1, 1);
		input <<25;
		Matrix Add(1, 1);
		Add <<50;
		Matrix Add2(1, 1);
		Add2 <<50;

		auto X = std::make_shared<Placeholder>(input,1);
		graph->setInput(X);

		auto sum = OperationsFactory::createSummationOp(graph,X,Add,1);
		auto sum2 = OperationsFactory::createSummationOp(graph,sum,Add2,1);
		graph->computeForward();

		Matrix test(1,1);
		test<<125;
		std::cout<<sum2->getForward()<<std::endl;
		REQUIRE(sum2->getForward().isApprox(test));
	}

}

/*
 * Gradients need no testing as it is only passing the Gradients to the Input Nodes
 */

