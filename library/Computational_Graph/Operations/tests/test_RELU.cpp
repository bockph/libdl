//
// Created by pbo on 17.06.19.
//

//
// Created by pbo on 17.06.19.
//

#include <Session.hpp>
#include <Placeholder.hpp>
#include <memory>

#include <iostream>

#include <catch2/catch.hpp>

#include <ReLuOp.hpp>


TEST_CASE("RELU Forward ", "[operation]") {



        Eigen::MatrixXf img(1, 3);
        img <<-1,0,1;


        auto X = std::make_shared<Placeholder>(img);

        auto conv = std::make_shared<ReLuOp>(X);

    Session session(conv);
        session.run();

        Eigen::MatrixXf test = Eigen::MatrixXf(1, 3);
        test <<0,0,1;

        REQUIRE(conv->getForward().isApprox(test)
        );


}
TEST_CASE("RELU Backward ", "[operation]") {



     Eigen::MatrixXf img(1, 3);
    img <<-1,0,5;


    auto X = std::make_shared<Placeholder>(img);

    auto conv = std::make_shared<ReLuOp>(X);

    Session session(conv);
    session.run();

    Eigen::MatrixXf test = Eigen::MatrixXf(1, 3);
    test <<1,1,1;
    REQUIRE(X->getCurrentGradients().isApprox(test)
    );
    //TODO: A test is missing for incoming negative gradients


}


