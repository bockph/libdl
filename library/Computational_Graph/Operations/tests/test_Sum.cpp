//
// Created by phili on 14.05.2019.
//
#include <SummationOp.hpp>
#include <Session.hpp>
#include <Placeholder.hpp>
#include <memory>

#include <catch2/catch.hpp>



/*TEST_CASE("SUM ", "[operation]") {
	auto x1 = std::make_shared<Placeholder>(25);
	auto x2 = std::make_shared<Placeholder>(50);
	auto x3 = std::make_shared<Placeholder>(75);
	auto x4 = std::make_shared<Placeholder>(100);
	auto a = std::make_shared<Placeholder>(2);
	auto b = std::make_shared<Placeholder>(3);
	auto c = std::make_shared<Placeholder>(4);
	auto graph = std::make_unique<Graph>();

	SECTION("Addition of Two") {
		std::vector<std::shared_ptr<Node>> forZ;
		forZ.push_back(x1);
		forZ.push_back(x2);
		auto o1 = std::make_shared<SUM>(forZ);
		Session session(o1, std::move(graph));
		session.run();
		REQUIRE(o1->getForwardData() == 75);
	}
	SECTION("Addition of more than two in one SumNode") {
		std::vector<std::shared_ptr<Node>> forZ;
		forZ.push_back(x1);
		forZ.push_back(x2);
		forZ.push_back(x3);
		forZ.push_back(x4);
		auto o1 = std::make_shared<SUM>(forZ);
		Session session(o1, std::move(graph));
		session.run();
		REQUIRE(o1->getForwardData() == 250);
	}
	SECTION("Changing a Nodes Value afterwards and doing Rerun") {
		std::vector<std::shared_ptr<Node>> forZ;
		forZ.push_back(x1);
		forZ.push_back(x2);
		auto o1 = std::make_shared<SUM>(forZ);
		Session session(o1, std::move(graph));
		session.run();
		REQUIRE(o1->getForwardData() == 75);
		*x1 = 75;
		session.run();
		REQUIRE(o1->getForwardData() == 125);
	}
	SECTION("Test correct gradient calculation with one Node"){
		std::vector<std::shared_ptr<Node>> forZ;
		forZ.push_back(a);
		forZ.push_back(b);
		forZ.push_back(c);
		auto o1 = std::make_shared<SUM>(forZ);
		Session session(o1, std::move(graph));
		session.run();
		REQUIRE(o1->getForwardData()==9);
		REQUIRE(o1->_gradients(0)==1);
		REQUIRE(o1->_gradients(1)==1);
		REQUIRE(o1->_gradients(2)==1);
	}
	SECTION("Test correct gradient calculation with multiple Nodes"){
		std::vector<std::shared_ptr<Node>> forZ;
		forZ.push_back(a);
		forZ.push_back(b);
//		forZ.push_back(c);
		auto o1 = std::make_shared<SUM>(forZ);
		std::vector<std::shared_ptr<Node>> sum2;
		sum2.push_back(o1);
		sum2.push_back(c);
		auto o2 = std::make_shared<SUM>(sum2);

		Session session(o2, std::move(graph));
		session.run();
		REQUIRE(o1->getForwardData()==5);
		REQUIRE(o2->getForwardData()==9);
		REQUIRE(o1->_gradients(0)==1);
		REQUIRE(o1->_gradients(1)==1);
		REQUIRE(o2->_gradients(0)==1);
	}

}

*/