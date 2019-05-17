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
		std::cout<<"forward1 0"<<":"<<o1->_forwardCache(0)<<std::endl;
		std::cout<<"forward1 1"<<":"<<o1->_forwardCache(1)<<std::endl;
		std::cout<<"forward2 0"<<":"<<o2->_forwardCache(0)<<std::endl;
		std::cout<<"forward2 1"<<":"<<o2->_forwardCache(1)<<std::endl;
		std::cout<<"forward3 0"<<":"<<o3->_forwardCache(0)<<std::endl;
		std::cout<<"forward3 1"<<":"<<o3->_forwardCache(1)<<std::endl;

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


