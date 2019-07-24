//
// Created by phili on 14.06.2019.
//

//
// Created by phili on 15.05.2019.
//
//
// Created by phili on 14.05.2019.
//
#include <MultiplicationOp.hpp>
#include <Session.hpp>
#include <Placeholder.hpp>
#include <memory>

#include <catch2/catch.hpp>
#include <iostream>

#include <Weight.hpp>
#include <ConvolveFilterIM2COL.hpp>


TEST_CASE("Convolution of Weight ", "[operation]") {

	SECTION("One dimensional filter, one input", "[One_Channel_Image]") {

		Eigen::MatrixXf img(1, 4);
		img << 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;

		auto X = std::make_shared<Placeholder>(img, 2, 1);
		auto W = std::make_shared<Weight>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

		Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
		test << 2, 4, 6, 8;
		std::cout<<conv->getForward()<<std::endl;
		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[One_Channel_Image]") {

		Eigen::MatrixXf img(2, 4);
		img << 1, 2, 3, 4,
				1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 2, 1);
		auto W = std::make_shared<Weight>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

		Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(2, 4);
		test << 2, 4, 6, 8
				, 2, 4, 6, 8;
        std::cout<<conv->getForward()<<std::endl;
		REQUIRE(conv->getForward().isApprox(test));
	}
    SECTION("One dimensional filter, stride 2", "[One_Channel_Image]") {

        Eigen::MatrixXf img(1, 9);
        img << 1, 2, 3, 4, 5, 6, 7, 8, 9;

        Eigen::MatrixXf filter(1, 1);
        filter << 2;
        Eigen::MatrixXf filter2(1, 1);
        filter2 << 2;


        auto X = std::make_shared<Placeholder>(img, 3, 1);
        auto W = std::make_shared<Weight>(filter, 1, 1);
        auto W2 = std::make_shared<Weight>(filter2, 1, 1);

        auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W, 2);

        Session session(conv);
        session.run();
        Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
        test << 2, 6, 14, 18;

        REQUIRE(conv->getForward().isApprox(test));

    }
    SECTION("One dimensional filter, stride 3", "[One_Channel_Image]") {

        Eigen::MatrixXf img(1, 9);
        img << 1, 2, 3, 4, 5, 6, 7, 8, 9;

        Eigen::MatrixXf filter(1, 1);
        filter << 2;



        auto X = std::make_shared<Placeholder>(img, 3, 1);
        auto W = std::make_shared<Weight>(filter, 1, 1);



        auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W, 3);

        Session session2(conv);
        session2.run();
        Eigen::MatrixXf test2 = Eigen::MatrixXf(1, 1);
        test2 << 2;
        std::cout<<conv->getForward()<<std::endl;
        REQUIRE(conv->getForward().isApprox(test2));
    }
	SECTION("Multi-dimensional filter, stride 1", "[One_Channel_Image]") {

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
		auto W = std::make_shared<Weight>(filter, 3, 1);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

		Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 9);
		test <<
			 4, 3, 4,
				2, 4, 3,
				2, 3, 4;
		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("MultiChannel", "[Multi_Channel_Image]") {

		Eigen::MatrixXf img(1, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Weight>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

		Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
		test << 5, 10, 15, 20;

		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[Multi_Channel_Image]") {

		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;


		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Weight>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

        Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(2, 4);
		test << 5, 10, 15, 20,
				5, 10, 15, 20;

		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("Two Filters and Two Convolutional Layers", "[Multi_Channel_Image]") {

		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(2, 2);
		filter << 2, 3,
				1, 1;
		Eigen::MatrixXf filter2(1, 2);
		filter2 << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Weight>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);
		auto W2 = std::make_shared<Weight>(filter2, 1, 2);
		auto conv2 = std::make_shared<ConvolveFilterIM2COL>(conv, W2);


//		Session session(conv, std::move(graph));
//		session.run();



        Session session2(conv2);
		session2.run();
		//Test ForwardPass
        Eigen::MatrixXf testConvF = Eigen::MatrixXf(2, 8);
        testConvF << 5, 10, 15, 20, 2, 4, 6, 8,
                5, 10, 15, 20, 2, 4, 6, 8;
        std::cout<<"actual"<<conv->getForward()<<std::endl;
        REQUIRE(conv->getForward().isApprox(testConvF));


		Eigen::MatrixXf testConv2F = Eigen::MatrixXf(2, 4);
        testConv2F << 16, 32, 48, 64,
				16, 32, 48, 64;
		REQUIRE(conv2->getForward().isApprox(testConv2F));

		//Test Backwardpass Weights


        Eigen::MatrixXf testW = Eigen::MatrixXf(2, 2);
        testW << 20, 20, 30, 30;
//        REQUIRE(W->getCurrentGradients().isApprox(testW));




        Eigen::MatrixXf testW2= Eigen::MatrixXf(1, 2);
        testW2 << 50,20;
        std::cout<<W->getCurrentGradients()<<std::endl;
        std::cout<<W2->getCurrentGradients()<<std::endl;
//        REQUIRE(W2->getCurrentGradients().isApprox(testW2));

        //Test Backwardpass Placeholder

        Eigen::MatrixXf testConvG = Eigen::MatrixXf(2, 8);
        testConvG << 	2,2,2,2,3,3,3,3,
                2,2,2,2,3,3,3,3;
        Eigen::MatrixXf testCG = Eigen::MatrixXf(2, 8);
        testCG << 7,7,7,7,9,9,9,9,
                7,7,7,7,9,9,9,9;
        REQUIRE(conv->getCurrentGradients().isApprox(testConvG));
        REQUIRE(X->getCurrentGradients().isApprox(testCG));

	}


};


TEST_CASE("Backpropagation Weight ", "[operation]") {

	SECTION("One dimensional filter, one input", "[One_Channel_Image]") {

		Eigen::MatrixXf img(1, 4);
		img << 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;

		auto X = std::make_shared<Placeholder>(img, 2, 1);
		auto W = std::make_shared<Weight>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

        Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 1);
		test << 10;
//        std::cout<<W->getCurrentGradients()<<std::endl;

		REQUIRE(W->getCurrentGradients().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[One_Channel_Image]") {

		Eigen::MatrixXf img(2, 4);
		img << 1, 2, 3, 4,
				1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 2, 1);
		auto W = std::make_shared<Weight>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

        Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 1);
		test << 20;
        std::cout<<W->getCurrentGradients()<<std::endl;
		REQUIRE(W->getCurrentGradients().isApprox(test));
	}
	SECTION("One dimensional filter, stride >1", "[One_Channel_Image]") {

		Eigen::MatrixXf img(1, 9);
		img << 1, 2, 3, 4, 5, 6, 7, 8, 9;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;
        Eigen::MatrixXf filter2(1, 1);
        filter2 << 2;




		auto X = std::make_shared<Placeholder>(img, 3, 1);
		auto W = std::make_shared<Weight>(filter, 1, 1);
        auto W2 = std::make_shared<Weight>(filter2, 1, 1);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W, 2);

        Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 1);
		test << 1 + 3 + 7 + 9;

		REQUIRE(W->getCurrentGradients().isApprox(test));

		auto conv2 = std::make_shared<ConvolveFilterIM2COL>(X, W2, 3);

        Session session2(conv2);
		session2.run();
		Eigen::MatrixXf test2 = Eigen::MatrixXf(1, 1);
		test2 << 1;
        std::cout<< "hello:\n"<<  W2->getCurrentGradients()<<std::endl;
		REQUIRE(W2->getCurrentGradients().isApprox(test2));
	}
	SECTION("Multi-dimensional filter, stride 1", "[One_Channel_Image]") {

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
		auto W = std::make_shared<Weight>(filter, 3, 1);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

        Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 9);
		test <<
			 6, 7, 6,
				4, 7, 7,
				4, 6, 6;
		REQUIRE(W->getCurrentGradients().isApprox(test));
	}
	SECTION("MultiChannel", "[Multi_Channel_Image]") {

		Eigen::MatrixXf img(1, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Weight>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

        Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 2);
		test << 10, 10;

		REQUIRE(W->getCurrentGradients().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[Multi_Channel_Image]") {

		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;


		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Weight>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

        Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 2);
		test << 20, 20;
		REQUIRE(W->getCurrentGradients().isApprox(test));
	}
	SECTION("Two Filters and Two Convolutional Layers", "[Multi_Channel_Image]") {

		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 2, 4, 6, 8,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(2, 2);
		filter << 2, 3,
				1, 1;
		Eigen::MatrixXf filter2(1, 2);
		filter2 << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Weight>(filter, 1, 2);
		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);
		auto W2 = std::make_shared<Weight>(filter2, 1, 2);
		auto conv2 = std::make_shared<ConvolveFilterIM2COL>(conv, W2);

        Session session2(conv2);
		session2.run();





	}


};


TEST_CASE("Backpropagation Input ", "[operation]") {

	SECTION("One dimensional filter, one input", "[One_Channel_Image]") {

		Eigen::MatrixXf img(1, 4);
		img << 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;

		auto X = std::make_shared<Placeholder>(img, 2, 1);
		auto W = std::make_shared<Weight>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

        Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
		test << 2, 2, 2, 2;

		REQUIRE(X->getCurrentGradients().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[One_Channel_Image]") {

		Eigen::MatrixXf img(2, 4);
		img << 1, 2, 3, 4,
				1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 2, 1);
		auto W = std::make_shared<Weight>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

        Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(2, 4);
		test << 2, 2, 2, 2,
				2, 2, 2, 2;

		std::cout << X->getCurrentGradients() << std::endl;
		REQUIRE(X->getCurrentGradients().isApprox(test));
	}
	SECTION("One dimensional filter, stride >1", "[One_Channel_Image]") {

 		Eigen::MatrixXf img(1, 9);
		img << 1, 2, 3, 4, 5, 6, 7, 8, 9;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;
        Eigen::MatrixXf filter2(1, 1);
        filter2 << 2;



        auto W2 = std::make_shared<Weight>(filter2, 1, 1);


		auto X = std::make_shared<Placeholder>(img, 3, 1);
		auto W = std::make_shared<Weight>(filter, 1, 1);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W, 2);

        Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1,9);
		test << 2,0,2,
				0,0,0,
				2,0,2;

		std::cout<<X->getCurrentGradients()<<std::endl;
        std::cout<<test<<std::endl;

        REQUIRE(X->getCurrentGradients().isApprox(test));

		auto conv2 = std::make_shared<ConvolveFilterIM2COL>(X, W2, 3);

		Session session2(conv2 );
		session2.run();
		Eigen::MatrixXf test2 = Eigen::MatrixXf(1, 1);
		test2 << 1;

		REQUIRE(W2->getCurrentGradients().isApprox(test2));
	}
	SECTION("Multi-dimensional filter, stride 1", "[One_Channel_Image]") {

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
		auto W = std::make_shared<Weight>(filter, 3, 1);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

        Session session(conv);
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

 		Eigen::MatrixXf img(1, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Weight>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

        Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 8);
		test << 2,2,2,2,3,3,3,3;
		std::cout<<X->getCurrentGradients()<<std::endl;

		REQUIRE(X->getCurrentGradients().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[Multi_Channel_Image]") {

 		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;


		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Weight>(filter, 1, 2);

		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);

        Session session(conv);
		session.run();
		Eigen::MatrixXf test = Eigen::MatrixXf(2, 8);
		test << 2,2,2,2,3,3,3,3,
				2,2,2,2,3,3,3,3;
		REQUIRE(X->getCurrentGradients().isApprox(test));
	}
	SECTION("Two Filters and Two Convolutional Layers", "[Multi_Channel_Image]") {
 		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(2, 2);
		filter << 2, 3,
				1, 1;
		Eigen::MatrixXf filter2(1, 2);
		filter2 << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2, 2);
		auto W = std::make_shared<Weight>(filter, 1, 2);
		auto conv = std::make_shared<ConvolveFilterIM2COL>(X, W);
		auto W2 = std::make_shared<Weight>(filter2, 1, 2);
		auto conv2 = std::make_shared<ConvolveFilterIM2COL>(conv, W2);

		Session session2(conv2);
		session2.run();
	}


};



