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
#include <Weight.hpp>
#include <iostream>
#include <SigmoidOP.hpp>
#include <MSEOp.hpp>
#include <SummationOp.hpp>
#include <Bias.hpp>


TEST_CASE("Multiplication Node ", "[operation]") {
	/*auto x1 = std::make_shared<Placeholder>(25);
	auto x2 = std::make_shared<Placeholder>(50);
	auto x3 = std::make_shared<Placeholder>(75);
	auto x4 = std::make_shared<Placeholder>(100);
	auto a = std::make_shared<Placeholder>(2);
	auto b = std::make_shared<Placeholder>(3);
	auto c = std::make_shared<Placeholder>(4);
	auto graph = std::make_unique<Graph>();

	SECTION("Multiplication of Two") {
		std::vector<std::shared_ptr<Node>> forZ;
		forZ.push_back(x1);
		forZ.push_back(x2);
		auto o1 = std::make_shared<MultiplicationOp>(forZ);
		Session session(o1, std::move(graph));
		session.run();
		REQUIRE(o1->getForwardData() == 1250);
	}
	SECTION("Multiplication of more than two in one SumNode") {
		std::vector<std::shared_ptr<Node>> forZ;
		forZ.push_back(x1);
		forZ.push_back(x2);
		forZ.push_back(x3);
		forZ.push_back(x4);
		auto o1 = std::make_shared<MultiplicationOp>(forZ);
		Session session(o1, std::move(graph));
		session.run();
		REQUIRE(o1->getForwardData() == 9375000);
	}
	SECTION("Changing a Nodes Value afterwards and doing Rerun") {
		std::vector<std::shared_ptr<Node>> forZ;
		forZ.push_back(x1);
		forZ.push_back(x2);
		auto o1 = std::make_shared<MultiplicationOp>(forZ);
		Session session(o1, std::move(graph));
		session.run();
		REQUIRE(o1->getForwardData() == 1250);
		*x1 = 75;
		session.run();
		REQUIRE(o1->getForwardData() == 3750);
	}
	SECTION("Test correct gradient calculation"){
		std::vector<std::shared_ptr<Node>> forZ;
		forZ.push_back(a);
		forZ.push_back(b);
		forZ.push_back(c);
		auto o1 = std::make_shared<MultiplicationOp>(forZ);
		Session session(o1, std::move(graph));
		session.run();
		REQUIRE(o1->getForwardData()==24);
		REQUIRE(o1->_gradients(0)==12);
		REQUIRE(o1->_gradients(1)==8);
		REQUIRE(o1->_gradients(2)==6);
	}*/
	SECTION("without activation functions and one input vector","[XOR Computational Graph]"){
		Eigen::MatrixXf mX1(1,2);
		mX1<<2,	10;

		Eigen::MatrixXf mW1(2,2);
		mW1<<1,3,
				2,4;

		Eigen::MatrixXf mW2(2,1);
		mW2<<6,7;


		auto X = std::make_shared<Placeholder>(mX1);
		auto W = std::make_shared<Weight>(mW1);
		auto W2 = std::make_shared<Weight>(mW2);

		auto mul = std::make_shared<MultiplicationOp>(X,W);
		auto mul2 = std::make_shared<MultiplicationOp>(mul,W2);

		Session session(mul2);
		session.run();
		REQUIRE(mul->getForward()(0,0)==22);
		REQUIRE(mul->getForward()(0,1)==46);
		REQUIRE(mul2->getForward()(0,0)==454);

		REQUIRE(W2->getCurrentGradients()(0,0)==22);
		REQUIRE(W2->getCurrentGradients()(1,0)==46);

		REQUIRE(X->getCurrentGradients()(0,0)==27);
		REQUIRE(X->getCurrentGradients()(0,1)==40);

		REQUIRE(W->getCurrentGradients()(0,0)==12);
		REQUIRE(W->getCurrentGradients()(0,1)==14);
		REQUIRE(W->getCurrentGradients()(1,0)==60);
		REQUIRE(W->getCurrentGradients()(1,1)==70 );
	}
	SECTION("without activation functions and 2 input vectors","[XOR Computational Graph]"){
		Eigen::MatrixXf mX1(2,2);
		mX1<<2,	10,4,20;

		Eigen::MatrixXf mW1(2,2);
		mW1<<1,3,
				2,4;

		Eigen::MatrixXf mW2(2,1);
		mW2<<6,7;



		auto X = std::make_shared<Placeholder>(mX1);
		auto W = std::make_shared<Weight>(mW1);
		auto W2 = std::make_shared<Weight>(mW2);

		auto mul = std::make_shared<MultiplicationOp>(X,W);
		auto mul2 = std::make_shared<MultiplicationOp>(mul,W2);

		Session session(mul2);
		session.run();
		REQUIRE(mul->getForward()(0,0)==22);
		REQUIRE(mul->getForward()(0,1)==46);
		REQUIRE(mul->getForward()(1,0)==44);
		REQUIRE(mul->getForward()(1,1)==92);
		REQUIRE(mul2->getForward()(0,0)==454);
		REQUIRE(mul2->getForward()(1,0)==908);


		REQUIRE(W2->getCurrentGradients()(0,0)==66);
		REQUIRE(W2->getCurrentGradients()(1,0)==138);

		REQUIRE(X->getCurrentGradients()(0,0)==27);
		REQUIRE(X->getCurrentGradients()(0,1)==40);
		REQUIRE(X->getCurrentGradients()(1,0)==27);
		REQUIRE(X->getCurrentGradients()(1,1)==40);

		REQUIRE(W->getCurrentGradients()(0,0)==36);
		REQUIRE(W->getCurrentGradients()(0,1)==42);
		REQUIRE(W->getCurrentGradients()(1,0)==180);
		REQUIRE(W->getCurrentGradients()(1,1)==210 );
	}

	SECTION("Test XOR Computational Graph withou activation functions"){

		Eigen::MatrixXf mX1(4,2);
		mX1<<
		   1,0,
				0,1,
				0,0,
				1,1;

		Eigen::MatrixXf mW1(2,2);
		mW1<<20,20,-20,-20;

		Eigen::MatrixXf mW2(2,1);
		mW2<<20,20;

		Eigen::MatrixXf C (4,1);
		C<<
		 1,
				1,
				0,
				0;


		auto X = std::make_shared<Placeholder>(mX1);
		auto W = std::make_shared<Weight>(mW1);
		auto W2 = std::make_shared<Weight>(mW2);

		auto mul = std::make_shared<MultiplicationOp>(X,W);
		auto sig1 = std::make_shared<SigmoidOP>(mul);
		auto mul2 = std::make_shared<MultiplicationOp>(mul,W2);
		auto sig2 = std::make_shared<SigmoidOP>(mul2);
		auto CN = std::make_shared<Placeholder>(C);
		auto mse =std::make_shared<MSEOp>(sig2,CN);

		/*Session session(mse);
		for(int i =0;i<1;i++) {
			session.run();
			std::cout << "         Round " << i << std::endl;
			std::cout << "Output:\n" << sig2->getForward() << std::endl;
			std::cout << "LOSS:\n" << mse->getForward() << std::endl;
		}*/
	}

	SECTION("Test XOR Computational Graph withou activation functions"){

		Eigen::MatrixXf mX1(4,2);
		mX1<<
		1,0,
		0,1,
		0,0,
		1,1;
		Eigen::MatrixXf C (4,1);
		C<<
		 1,
				1,
				0,
				0;
		auto CN = std::make_shared<Placeholder>(C);

		Eigen::MatrixXf mW1(2,2);
		mW1<<0.5,0.4,0.3,0.234;//,0.3,0.7;
		Eigen::MatrixXf b1(1,2);
		b1<<1,1;//,1,1,1,1,1,1;

		Eigen::MatrixXf mW12(2,2);
		mW12<<0.8,0.3,0.7,0.3;

		Eigen::MatrixXf mW2(2,1);
		mW2<<0.5,0.1;//,0.4,0.8;
		Eigen::MatrixXf b2(1,1);
		b2<<1;//,1,1,1;



		auto X = std::make_shared<Placeholder>(mX1);
		auto W = std::make_shared<Weight>(mW1);
		auto B1 = std::make_shared<Bias>(b1);
		auto B2 = std::make_shared<Bias>(b2);
//		auto W12 = std::make_shared<Weight>(mW12);

		auto W2 = std::make_shared<Weight>(mW2);

		auto mul = std::make_shared<MultiplicationOp>(X,W);
		auto sum = std::make_shared<SummationOp>(mul,B1);
		auto sig1 = std::make_shared<SigmoidOP>(sum);
//		auto mul1 = std::make_shared<MultiplicationOp>(sig1,W12);
//		auto sig12 = std::make_shared<SigmoidOP>(mul1);
		auto mul2 = std::make_shared<MultiplicationOp>(sig1,W2);
		auto sum2 = std::make_shared<SummationOp>(mul2,B2);


		auto sig2 = std::make_shared<SigmoidOP>(sum2);
		auto mse =std::make_shared<MSEOp>(sig2,CN);

		Session session(mse);
		session.run();
		std::cout<<"         Round "<<1<<std::endl;
		std::cout<<"Output:\n"<<sig2->getForward()<<std::endl;
		std::cout<<"LOSS:\n"<<mse->getForward()<<std::endl;
		for(int i =0;i<500;i++){
			session.run();
//			std::cout<<"         Round "<<i<<std::endl;
//			std::cout<<"Output:\n"<<sig2->getForward()<<std::endl;
//			std::cout<<"LOSS:\n"<<mse->getForward()<<std::endl;

//		std::cout<<"        Gradientx1"<<nMul31->_gradients(0)<<std::endl;

		}
		session.run();
		std::cout<<"         Round "<<100001<<std::endl;
		std::cout<<"Output:\n"<<sig2->getForward()<<std::endl;
		std::cout<<"LOSS:\n"<<mse->getForward()<<std::endl;
		/*REQUIRE(mul->getForward()(0,0)==22);
		REQUIRE(mul->getForward()(0,1)==46);
		REQUIRE(mul2->getForward()(0,0)==454);

		REQUIRE(W2->getCurrentGradients()(0,0)==22);
		REQUIRE(W2->getCurrentGradients()(1,0)==46);

		REQUIRE(X->getCurrentGradients()(0,0)==27);
		REQUIRE(X->getCurrentGradients()(0,1)==40);
		REQUIRE(W->getCurrentGradients()(0,0)==12);
		REQUIRE(W->getCurrentGradients()(0,1)==14);
		REQUIRE(W->getCurrentGradients()(1,0)==60);
		REQUIRE(W->getCurrentGradients()(1,1)==70 );*/
	}

}


