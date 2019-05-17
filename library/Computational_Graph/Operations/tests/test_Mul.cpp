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


TEST_CASE("Multiplication Node ", "[operation]") {
	auto x1 = std::make_shared<Placeholder>(25);
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
		auto o1 = std::make_shared<MUL>(forZ);
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
		auto o1 = std::make_shared<MUL>(forZ);
		Session session(o1, std::move(graph));
		session.run();
		REQUIRE(o1->getForwardData() == 9375000);
	}
	SECTION("Changing a Nodes Value afterwards and doing Rerun") {
		std::vector<std::shared_ptr<Node>> forZ;
		forZ.push_back(x1);
		forZ.push_back(x2);
		auto o1 = std::make_shared<MUL>(forZ);
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
		auto o1 = std::make_shared<MUL>(forZ);
		Session session(o1, std::move(graph));
		session.run();
		REQUIRE(o1->getForwardData()==24);
		REQUIRE(o1->_gradients(0)==12);
		REQUIRE(o1->_gradients(1)==8);
		REQUIRE(o1->_gradients(2)==6);
	}


}


