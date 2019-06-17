//
// Created by pbo on 17.06.19.
//




#include <MaxPool.hpp>


TEST_CASE("Convolution of Filter ", "[operation]") {

SECTION("Multi-dimensional filter, stride 1", "[One_Channel_Image]") {

auto graph = std::make_unique<Graph>();
Eigen::MatrixXf img(1, 25);
img <<
1,2,1,4,
0,0,3,0,
1,2,0,0,
0,0,0,0;


auto X = std::make_shared<Placeholder>(img);

auto conv = std::make_shared<MaxPool>(X, 2,2);

Session session(conv, std::move(graph));
session.

run();

Eigen::MatrixXf test = Eigen::MatrixXf(1, 9);
test <<
2,4,
2,0;
std::cout<<"Result: "<<conv->

getForward()

<<
std::endl;

REQUIRE(conv
->

getForward()

.
isApprox(test)
);
}

}


