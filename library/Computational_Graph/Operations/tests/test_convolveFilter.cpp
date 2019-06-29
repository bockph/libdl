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

		auto X = std::make_shared<Placeholder>(img, 2, 1);
		auto W = std::make_shared<Filter>(filter, 1, 1);

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


		auto X = std::make_shared<Placeholder>(img, 2, 1);
		auto W = std::make_shared<Filter>(filter, 1, 1);

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
		img << 1, 2, 3, 4, 5, 6, 7, 8, 9;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 3, 1);
		auto W = std::make_shared<Filter>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilter>(X, W, 2);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
		test << 2, 6, 14, 18;

		REQUIRE(conv->getForward().isApprox(test));

		auto conv2 = std::make_shared<ConvolveFilter>(X, W, 3);

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
			1, 1, 1, 0, 0,
				0, 1, 1, 1, 0,
				0, 0, 1, 1, 1,
				0, 0, 1, 1, 0,
				0, 1, 1, 0, 0;
		Eigen::MatrixXf filter(1, 9);
		filter <<
			   1, 0, 1,
				0, 1, 0,
				1, 0, 1;

		auto X = std::make_shared<Placeholder>(img, 5, 1);
		auto W = std::make_shared<Filter>(filter, 3, 1);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 9);
		test <<
			 4, 3, 4,
				2, 4, 3,
				2, 3, 4;
		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("MultiChannel", "[Multi_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(1, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Filter>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
		test << 5, 10, 15, 20;

		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[Multi_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;


		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Filter>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(2, 4);
		test << 5, 10, 15, 20,
				5, 10, 15, 20;

		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("Two Filters and Two Convolutional Layers", "[Multi_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(2, 2);
		filter << 2, 3,
				1, 1;
		Eigen::MatrixXf filter2(1, 2);
		filter2 << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Filter>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilter>(X, W);
		auto W2 = std::make_shared<Filter>(filter2, 1, 2);
		auto conv2 = std::make_shared<ConvolveFilter>(conv, W2);


		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(2, 8);
		test << 5, 10, 15, 20, 2, 4, 6, 8,
				5, 10, 15, 20, 2, 4, 6, 8;

		REQUIRE(conv->getForward().isApprox(test));


		Session session2(conv2, std::move(graph));
		session2.run();
		Eigen::MatrixXf test2 = Eigen::MatrixXf(2, 4);
		test2 << 16, 32, 48, 64,
				16, 32, 48, 64;
		REQUIRE(conv2->getForward().isApprox(test2));


	}


};


TEST_CASE("Backpropagation Filter ", "[operation]") {

	SECTION("One dimensional filter, one input", "[One_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(1, 4);
		img << 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;

		auto X = std::make_shared<Placeholder>(img, 2, 1);
		auto W = std::make_shared<Filter>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 1);
		test << 10;
//        std::cout<<W->getCurrentGradients()<<std::endl;

		REQUIRE(W->getCurrentGradients().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[One_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(2, 4);
		img << 1, 2, 3, 4,
				1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 2, 1);
		auto W = std::make_shared<Filter>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 1);
		test << 10;

		REQUIRE(W->getCurrentGradients().isApprox(test));
	}
	SECTION("One dimensional filter, stride >1", "[One_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(1, 9);
		img << 1, 2, 3, 4, 5, 6, 7, 8, 9;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 3, 1);
		auto W = std::make_shared<Filter>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilter>(X, W, 2);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 1);
		test << 1 + 3 + 7 + 9;

		REQUIRE(W->getCurrentGradients().isApprox(test));

		auto conv2 = std::make_shared<ConvolveFilter>(X, W, 3);

		Session session2(conv2, std::move(graph));
		session2.run();
		Eigen::MatrixXf test2 = Eigen::MatrixXf(1, 1);
		test2 << 1;

		REQUIRE(W->getCurrentGradients().isApprox(test2));
	}
	SECTION("Multi-dimensional filter, stride 1", "[One_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(1, 25);
		img <<
			1, 1, 1, 0, 0,
				0, 1, 1, 1, 0,
				0, 0, 1, 1, 1,
				0, 0, 1, 1, 0,
				0, 1, 1, 0, 0;
		Eigen::MatrixXf filter(1, 9);
		filter <<
			   1, 0, 1,
				0, 1, 0,
				1, 0, 1;

		auto X = std::make_shared<Placeholder>(img, 5, 1);
		auto W = std::make_shared<Filter>(filter, 3, 1);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 9);
		test <<
			 6, 7, 6,
				4, 7, 7,
				4, 6, 6;
		REQUIRE(W->getCurrentGradients().isApprox(test));
	}
	SECTION("MultiChannel", "[Multi_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(1, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Filter>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 2);
		test << 10, 10;

		REQUIRE(W->getCurrentGradients().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[Multi_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;


		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Filter>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 2);
		test << 10, 10;
		REQUIRE(W->getCurrentGradients().isApprox(test));
	}
	SECTION("Two Filters and Two Convolutional Layers", "[Multi_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 2, 4, 6, 8,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(2, 2);
		filter << 2, 3,
				1, 1;
		Eigen::MatrixXf filter2(1, 2);
		filter2 << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Filter>(filter, 1, 2);
		auto conv = std::make_shared<ConvolveFilter>(X, W);
		auto W2 = std::make_shared<Filter>(filter2, 1, 2);
		auto conv2 = std::make_shared<ConvolveFilter>(conv, W2);

		Session session2(conv2, std::move(graph));
		session2.run();



		Eigen::MatrixXf test2 = Eigen::MatrixXf(1, 2);
		test2 << 65, 25;
        std::cout<<W->getCurrentGradients()<<std::endl;
		Eigen::MatrixXf test3 = Eigen::MatrixXf(2, 2);
		test3 << 20, 30, 30, 45;
		REQUIRE(W2->getCurrentGradients().isApprox(test2));
        REQUIRE(W->getCurrentGradients().isApprox(test3));


	}


};


TEST_CASE("Backpropagation Input ", "[operation]") {

	SECTION("One dimensional filter, one input", "[One_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(1, 4);
		img << 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;

		auto X = std::make_shared<Placeholder>(img, 2, 1);
		auto W = std::make_shared<Filter>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
		test << 2, 2, 2, 2;

		REQUIRE(X->getCurrentGradients().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[One_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(2, 4);
		img << 1, 2, 3, 4,
				1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 2, 1);
		auto W = std::make_shared<Filter>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(2, 4);
		test << 2, 2, 2, 2,
				2, 2, 2, 2;

		std::cout << X->getCurrentGradients() << std::endl;
		REQUIRE(X->getCurrentGradients().isApprox(test));
	}
	SECTION("One dimensional filter, stride >1", "[One_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(1, 9);
		img << 1, 2, 3, 4, 5, 6, 7, 8, 9;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 3, 1);
		auto W = std::make_shared<Filter>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilter>(X, W, 2);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1,9);
		test << 2,0,2,
				0,0,0,
				2,0,2;

		REQUIRE(X->getCurrentGradients().isApprox(test));

		auto conv2 = std::make_shared<ConvolveFilter>(X, W, 3);

		Session session2(conv2, std::move(graph));
		session2.run();
		Eigen::MatrixXf test2 = Eigen::MatrixXf(1, 1);
		test2 << 1;

		REQUIRE(W->getCurrentGradients().isApprox(test2));
	}
	SECTION("Multi-dimensional filter, stride 1", "[One_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(1, 25);
		img <<
				1, 1, 1, 0, 0,
				0, 1, 1, 1, 0,
				0, 0, 1, 1, 1,
				0, 0, 1, 1, 0,
				0, 1, 1, 0, 0;
		Eigen::MatrixXf filter(1, 9);
		filter <<
			   	1, 0, 1,
				0, 1, 0,
				1, 0, 1;

		auto X = std::make_shared<Placeholder>(img, 5, 1);
		auto W = std::make_shared<Filter>(filter, 3, 1);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 25);
		test <<
				1,1,2,1,1,
				1,2,3,2,1,
				2,3,5,3,2,
				1,2,3,2,1,
				1,1,2,1,1;
		REQUIRE(X->getCurrentGradients().isApprox(test));
	}
	SECTION("MultiChannel", "[Multi_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(1, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Filter>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 8);
		test << 2,2,2,2,3,3,3,3;
		std::cout<<X->getCurrentGradients()<<std::endl;

		REQUIRE(X->getCurrentGradients().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[Multi_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;


		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Filter>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilter>(X, W);

		Session session(conv, std::move(graph));
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(2, 8);
		test << 2,2,2,2,3,3,3,3,
				2,2,2,2,3,3,3,3;
		REQUIRE(X->getCurrentGradients().isApprox(test));
	}
	SECTION("Two Filters and Two Convolutional Layers", "[Multi_Channel_Image]") {

		auto graph = std::make_unique<Graph>();
		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(2, 2);
		filter << 2, 3,
				1, 1;
		Eigen::MatrixXf filter2(1, 2);
		filter2 << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Filter>(filter, 1, 2);
		auto conv = std::make_shared<ConvolveFilter>(X, W);
		auto W2 = std::make_shared<Filter>(filter2, 1, 2);
		auto conv2 = std::make_shared<ConvolveFilter>(conv, W2);

		Session session2(conv2, std::move(graph));
		session2.run();

		Eigen::MatrixXf test2 = Eigen::MatrixXf(2, 8);
		test2 << 	2,2,2,2,3,3,3,3,
					2,2,2,2,3,3,3,3;
		Eigen::MatrixXf test3 = Eigen::MatrixXf(2, 8);
		test3 << 7,7,7,7,9,9,9,9,
				7,7,7,7,9,9,9,9;
		REQUIRE(conv->getCurrentGradients().isApprox(test2));
        REQUIRE(X->getCurrentGradients().isApprox(test3));


	}


};



