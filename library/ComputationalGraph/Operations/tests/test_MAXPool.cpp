//
// Created by pbo on 17.06.19.
//



#include <Placeholder.hpp>
#include <memory>

#include <iostream>

#include <catch2/catch.hpp>

#include <MaxPoolOp.hpp>
#include <Graph.hpp>
#include <OperationsFactory.hpp>

TEST_CASE("Maxpool Forwardpass ", "[operation]") {

	auto graph = std::make_shared<Graph>();



    Matrix img(1, 16);
    img <<
        	1, 2, 1, 4,
            0, 0, 3, 0,
            1, 2, 0, 9,
            0, 0, 0, 0;


    auto X = std::make_shared<Placeholder>(img,1);
	graph->setInput(X);
	std::shared_ptr<MaxPoolOp> maxPool = OperationsFactory::createMaxpoolOp(graph, X, 2,2);
	graph->computeForward();

    SECTION("general Functionality", "[One_Channel_Image]") {


        Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
        test <<	2, 4,
                2, 9;

        REQUIRE(maxPool->getForward().isApprox(test)
        );
    }
    SECTION("IndexMatrix", "[One_Channel_Image]") {


        Eigen::MatrixXf test = Eigen::MatrixXf(1, 16);
        test <<
             	0, 1, 0, 1,
                0, 0, 0, 0,
                0, 1, 0, 1,
                0, 0, 0, 0;

        REQUIRE(maxPool->getMaxIndexMatrix().isApprox(test));
    }

}


TEST_CASE("Maxpool Backwardpass ", "[operation]") {
	auto graph = std::make_shared<Graph>();
    Eigen::MatrixXf img(1, 16);
    img <<
        	1, 2, 1, 4,
            0, 0, 3, 0,
            1, 2, 0, 9,
            0, 0, 0, 0;


    auto X = std::make_shared<Placeholder>(img, 1);

	graph->setInput(X);
	std::shared_ptr<MaxPoolOp> maxPool = OperationsFactory::createMaxpoolOp(graph, X, 2,2);
	graph->computeForward();
	graph->computeBackwards();

    SECTION("IndexMatrix", "[One_Channel_Image]") {


        Eigen::MatrixXf test = Eigen::MatrixXf(1, 16);
        test <<
             	0, 1, 0, 1,
                0, 0, 0, 0,
                0, 1, 0, 1,
                0, 0, 0, 0;

        REQUIRE(X->getPreviousGradients().isApprox(test));
    }

}

