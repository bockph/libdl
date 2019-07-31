//
// Created by pbo on 17.06.19.
//

//
// Created by pbo on 17.06.19.
//

#include <Placeholder.hpp>
#include <memory>

#include <iostream>

#include <catch2/catch.hpp>

#include <ReLuOp.hpp>
#include <Graph.hpp>
#include <OperationsFactory.hpp>


TEST_CASE("RELU Forward ", "[operation]") {
	auto graph = std::make_shared<Graph>();



        Eigen::MatrixXf img(1, 3);
        img <<-1,0,1;


        auto X = std::make_shared<Placeholder>(img,1);
		graph->setInput(X);
		auto relu = OperationsFactory::createReLuOp(graph,X);
		graph->computeForward();


        Eigen::MatrixXf test = Eigen::MatrixXf(1, 3);
        test <<0,0,1;

        REQUIRE(relu->getForward().isApprox(test)
        );


}
TEST_CASE("RELU Backward ", "[operation]") {
	auto graph = std::make_shared<Graph>();

     Eigen::MatrixXf img(1, 3);
    img <<-1,0,5;


	auto X = std::make_shared<Placeholder>(img,1);
	graph->setInput(X);
	auto relu = OperationsFactory::createReLuOp(graph,X);
	graph->computeForward();
	graph->computeBackwards();
	graph->updateParameters();
    Eigen::MatrixXf test = Eigen::MatrixXf(1, 3);
    test <<1,1,1;
    REQUIRE(X->getPreviousGradients().isApprox(test)
    );


}


