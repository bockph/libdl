//
// Created by phili on 15.05.2019.
//
#include <MUL.hpp>
#include <SUM.hpp>
#include <Session.hpp>
#include <Placeholder.hpp>
#include <memory>
#include <Graph.hpp>

#include <catch2/catch.hpp>
#include <iostream>
#include <Sigmoid.hpp>
#include <MSE.hpp>
#include <Weight.hpp>


TEST_CASE("Multiplication Node ", "[operation]") {
	auto x1 = std::make_shared<Placeholder>(25);
	auto x2 = std::make_shared<Placeholder>(50);
	auto x3 = std::make_shared<Placeholder>(75);
	auto x4 = std::make_shared<Placeholder>(100);
	auto a = std::make_shared<Placeholder>(2);
	auto b = std::make_shared<Placeholder>(3);
	auto c = std::make_shared<Placeholder>(4);
	auto graph = std::make_unique<Graph>();

	std::vector<std::shared_ptr<Node>> mul1;
	mul1.push_back(x1);
	mul1.push_back(x2);
	auto o1 = std::make_shared<MUL>(mul1);

	std::vector<std::shared_ptr<Node>> sum1;
	sum1.push_back(o1);
	sum1.push_back(x3);
	auto o2 = std::make_shared<SUM>(sum1);

	std::vector<std::shared_ptr<Node>> mul2;
	mul2.push_back(o2);
	mul2.push_back(x4);
	auto o3 = std::make_shared<MUL>(mul2);

//	std::vector<std::shared_ptr<Node>> sigVec;
//	sigVec.push_back(o3);
//	auto o4= std::make_shared<Sigmoid>(sigVec);

	Session session(o3, std::move(graph));
	session.run();
	SECTION("Multiplication SUM Multiplication Forwardpass") {

		REQUIRE(o1->getForwardData() == 1250);
		REQUIRE(o2->getForwardData()==1325);
		REQUIRE(o3->getForwardData()==(1325*100));
	}
	SECTION("Multiplication SUM Multiplication BackProp") {
		REQUIRE(o1->_gradients(0) == 5000);
		REQUIRE(o1->_gradients(1) == 2500);
		REQUIRE(o2->_gradients(0)==100);
		REQUIRE(o2->_gradients(1)==100);
		REQUIRE(o3->_gradients(0)==100);
		REQUIRE(o3->_gradients(1)==1325);
//		std::cout<<"Sigmoid:"<<o4->getForwardData()<<std::endl;
//		std::cout<<"Sigmoid gradient:"<<o4->_gradients(0)<<std::endl;
	}
//	SECTION("Multiplication of more than two in one SumNode") {
//		std::vector<std::shared_ptr<Node>> forZ;
//		forZ.push_back(x1);
//		forZ.push_back(x2);
//		forZ.push_back(x3);
//		forZ.push_back(x4);
//		auto o1 = std::make_shared<MUL>(forZ);
//		Session session(o1, std::move(graph));
//		session.run();
//		REQUIRE(o1->getForwardData() == 9375000);
//	}
//	SECTION("Changing a Nodes Value afterwards and doing Rerun") {
//		std::vector<std::shared_ptr<Node>> forZ;
//		forZ.push_back(x1);
//		forZ.push_back(x2);
//		auto o1 = std::make_shared<MUL>(forZ);
//		Session session(o1, std::move(graph));
//		session.run();
//		REQUIRE(o1->getForwardData() == 1250);
//		*x1 = 75;
//		session.run();
//		REQUIRE(o1->getForwardData() == 3750);
//	}
//	SECTION("Test correct gradient calculation"){
//		std::vector<std::shared_ptr<Node>> forZ;
//		forZ.push_back(a);
//		forZ.push_back(b);
//		forZ.push_back(c);
//		auto o1 = std::make_shared<MUL>(forZ);
//		Session session(o1, std::move(graph));
//		session.run();
//		REQUIRE(o1->getForwardData()==24);
//		REQUIRE(o1->_gradients(0)==12);
//		REQUIRE(o1->_gradients(1)==8);
//		REQUIRE(o1->_gradients(2)==6);
//	}


}
TEST_CASE("XOR test ", "[real problem]") {


	auto x1 = std::make_shared<Placeholder>(0);
	auto x2 = std::make_shared<Placeholder>(1);
	auto w11 =std::make_shared<Weight>(0.5);
	auto w12 =std::make_shared<Weight>(0.3);
	auto w21 =std::make_shared<Weight>(0.43);
	auto w22 =std::make_shared<Weight>(0.234);
//	auto w11 =std::make_shared<Placeholder>(3.6216969);
//	auto w12 =std::make_shared<Placeholder>(5.79319846);
//	auto w21 =std::make_shared<Placeholder>(3.63346351);
//	auto w22 =std::make_shared<Placeholder>(5.85580433);


	std::vector<std::shared_ptr<Node>> mul11 = {w11,x1};
	std::vector<std::shared_ptr<Node>> mul12 = {w21,x2};

	std::vector<std::shared_ptr<Node>> mul21 = {w12,x1};
	std::vector<std::shared_ptr<Node>> mul22 = {w22,x2};

	auto nMul11 = std::make_shared<MUL>(mul11);
	auto nMul12 = std::make_shared<MUL>(mul12);
	auto nMul21 = std::make_shared<MUL>(mul21);
	auto nMul22 = std::make_shared<MUL>(mul22);

	std::vector<std::shared_ptr<Node>> sum1 = {nMul11,nMul12};
	std::vector<std::shared_ptr<Node>> sum2 = {nMul21,nMul22};
	auto nSum1 =std::make_shared<SUM>(sum1);
	auto nSum2 =std::make_shared<SUM>(sum2);


	std::vector<std::shared_ptr<Node>> sig1 = {nSum1};
	std::vector<std::shared_ptr<Node>> sig2 = {nSum2};
	auto nsig1 =std::make_shared<Sigmoid>(sig1);
	auto nsig2 =std::make_shared<Sigmoid>(sig2);

//	auto w31 =std::make_shared<Placeholder>(-8.06441097);
//	auto w32 =std::make_shared<Placeholder>(7.41088524);
	auto w31 =std::make_shared<Weight>(0.9);
	auto w32 =std::make_shared<Weight>(0.1);

	std::vector<std::shared_ptr<Node>> mul31 = {w31 ,nsig1};
	std::vector<std::shared_ptr<Node>> mul32 = {w32 ,nsig2};
	auto nMul31 = std::make_shared<MUL>(mul31);
	auto nMul32 = std::make_shared<MUL>(mul32);

	std::vector<std::shared_ptr<Node>> sum3 = {nMul31,nMul32};
	auto nSum3 =std::make_shared<SUM>(sum3);

	std::vector<std::shared_ptr<Node>> sig3 = {nSum3};
	auto nsig3 =std::make_shared<Sigmoid>(sig3);


	auto c = std::make_shared<Placeholder>(1);

	std::vector<std::shared_ptr<Node>> loss = {nsig3,c};

	auto nloss =std::make_shared<MSE>(loss);

	auto graph = std::make_unique<Graph>();


	Session session(nloss, std::move(graph));
	for(int i =0;i<10000;i++){
		session.run();
		std::cout<<"Round "<<i<<std::endl;
		std::cout<<"        Output:"<<nsig3->getForwardData()<<std::endl;
		std::cout<<"        LOSS:"<<nloss->getForwardData()<<std::endl;

//		std::cout<<"        Gradientx1"<<nMul31->_gradients(0)<<std::endl;

	}
	x1->setForwardData(0);
	x2->setForwardData(0);
	session.run();
	std::cout<<"        Output:"<<nsig3->getForwardData()<<std::endl;



}


