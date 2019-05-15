#include <iostream>
//#include <Eigen/Dense>
#include <Graph.hpp>
#include <Placeholder.hpp>
#include <Operation.hpp>
#include <SUM.hpp>
#include <Session.hpp>
//#include "spdlog/spdlog.h"
//using Eigen::MatrixXd;
int main()
{
//
Graph graph;

	auto x1= std::make_shared<Placeholder>(50) ;
	auto x2= std::make_shared<Placeholder>(25) ;
	auto x3= std::make_shared<Placeholder>(75) ;

	std::vector<std::shared_ptr<Node>> forZ;
	forZ.push_back(x1);
	forZ.push_back(x2);

	auto o1 = std::make_shared<SUM>(forZ);






	Session session(o1,std::make_unique<Graph>(graph));

	session.run();


	std::cout<<"hello:"<<o1->getDatavalue()<<std::endl;

}
//
//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
//#include <catch2/catch.hpp>
//unsigned int Factorial( unsigned int number ) {
//	return number <= 1 ? number : Factorial(number-1)*number;
//}
//
//TEST_CASE( "Factorials are computed", "[factorial]" ) {
//REQUIRE( Factorial(1) == 0 );
//REQUIRE( Factorial(2) == 2 );
//REQUIRE( Factorial(3) == 6 );
//REQUIRE( Factorial(10) == 3628800 );
//}
//
//#include <pybind11/pybind11.h>
//
//namespace py = pybind11;
//
//
//int add(int i, int j) {
//	return i + j;
//}
//
//PYBIND11_MODULE(example, m) {
//	m.doc() = "pybind11 example plugin"; // optional module docstring
//
//	m.def("add", &add, "A function which adds two numbers");
//}