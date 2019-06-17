//
// Created by phili on 14.06.2019.
//

//
// Created by phili on 15.05.2019.
//
//
// Created by phili on 14.05.2019.
//
#include <MUL.hpp>
#include <Session.hpp>
#include <Placeholder.hpp>
#include <memory>
#include <Graph.hpp>

#include <catch2/catch.hpp>
#include <iostream>

#include <Filter.hpp>
#include <ConvolveFilter.hpp>


TEST_CASE("Convolution of Filter ", "[operation]") {

	SECTION("One dimensional filter, one input", "[One_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(1, 4);
		img << 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;

//auto test1 =img;
//img = img.transpose();
//test1.resize(2,2);
//std::cout<<"test1:"<<test1<<std::endl;
//Eigen::MatrixXf test2 =test1.block(0,0,3,3);
//std::cout<<"test2_1:"<<test2.size()<<std::endl;
//
//test2.resize(1,9);
//std::cout<<"test2_2:"<<test2<<std::endl;
		auto X = std::make_shared<Placeholder>(img);
		auto W = std::make_shared<Filter>(filter);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
		test << 2, 4, 6, 8;
		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[One_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(2, 4);
		img << 1, 2, 3, 4,
				1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img);
		auto W = std::make_shared<Filter>(filter);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(2, 4);
		test << 2, 4, 6, 8
				, 2, 4, 6, 8;

		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("One dimensional filter, stride >1", "[One_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(1, 9);
		img << 1, 2, 3, 4,5,6,7,8,9;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img);
		auto W = std::make_shared<Filter>(filter);

		auto conv = std::make_shared<ConvolveFilter>(X, W,2);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
		test << 2,  6,14,18;

        REQUIRE(conv->getForward().isApprox(test));

		auto conv2 = std::make_shared<ConvolveFilter>(X, W,3);

		Session session2(conv2, std::move(graph));
		session2.run();
		Eigen::MatrixXf test2 = Eigen::MatrixXf(1, 1);
		test2 << 2;

        REQUIRE(conv2->getForward().isApprox(test2));
	}
	SECTION("Multi-dimensional filter, stride 1", "[One_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(1, 25);
		img <<
		1,1,1,0,0,
		0,1,1,1,0,
		0,0,1,1,1,
		0,0,1,1,0,
		0,1,1,0,0;
		Eigen::MatrixXf filter(1, 9);
		filter <<
		1,0,1,
		0,1,0,
		1,0,1;

		auto X = std::make_shared<Placeholder>(img);
		auto W = std::make_shared<Filter>(filter);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 9);
		test <<
		4,3,4,
		2,4,3,
		2,3,4;
        std::cout<<"Result: "<<conv->getForward()<<std::endl;

        REQUIRE(conv->getForward().isApprox(test));
	}

}


