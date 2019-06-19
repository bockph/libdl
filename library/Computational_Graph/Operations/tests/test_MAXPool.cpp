//
// Created by pbo on 17.06.19.
//

//
// Created by pbo on 17.06.19.
//

#include <Session.hpp>
#include <Placeholder.hpp>
#include <memory>
#include <Graph.hpp>

#include <iostream>

#include <catch2/catch.hpp>

#include <MaxPool.hpp>


TEST_CASE("Maxpool Forwardpass ", "[operation]") {
    auto graph = std::make_unique<Graph>();
    Eigen::MatrixXf img(1, 16);
    img <<
            1, 2, 1, 4,
            0, 0, 3, 0,
            1, 2, 0, 9,
            0, 0, 0, 0;


    auto X = std::make_shared<Placeholder>(img,4);

    auto maxPool = std::make_shared<MaxPool>(X, 2, 2);

    Session session(maxPool, std::move(graph));
    session.run();
    SECTION("general Functionality", "[One_Channel_Image]") {



        Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
        test <<
                2, 4,
                2, 9;

        REQUIRE(maxPool->getForward().isApprox(test)
        );
    }
    SECTION("IndexMatrix", "[One_Channel_Image]") {


        Eigen::MatrixXf test = Eigen::MatrixXf(1, 16);
        test <<
            0,1,0,1,
            0,0,0,0,
            0,1,0,1,
            0,0,0,0;
        auto tmp = maxPool->getMaxIndexMatrix();
        tmp.resize(4, 4);
        tmp.transposeInPlace();

        REQUIRE(maxPool->getMaxIndexMatrix().isApprox(test)
        );
    }
    //TODO there might be an issue if e.g. stride is one, that one index is for example maximum value for two windows, then ther might should be an average update?

}


TEST_CASE("Maxpool Backwardpass ", "[operation]") {
    auto graph = std::make_unique<Graph>();
    Eigen::MatrixXf img(1, 16);
    img <<
            1, 2, 1, 4,
            0, 0, 3, 0,
            1, 2, 0, 9,
            0, 0, 0, 0;


    auto X = std::make_shared<Placeholder>(img,4);

    auto maxPool = std::make_shared<MaxPool>(X, 2, 2);

    Session session(maxPool, std::move(graph));
    session.run();

    SECTION("IndexMatrix", "[One_Channel_Image]") {


        Eigen::MatrixXf test = Eigen::MatrixXf(1, 16);
        test <<
             0,1,0,1,
                0,0,0,0,
                0,1,0,1,
                0,0,0,0;
        auto tmp = maxPool->getMaxIndexMatrix();
        tmp.resize(4, 4);
        tmp.transposeInPlace();

        REQUIRE(X->getCurrentGradients().isApprox(test)
        );
    }
    //TODO there might be an issue if e.g. stride is one, that one index is for example maximum value for two windows, then ther might should be an average update?

}

