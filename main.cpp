#include <iostream>
//#include <Eigen/Dense>
#include <Graph.hpp>
#include <Placeholder.hpp>
#include <Operation.hpp>
#include <SUM.hpp>
#include <Session.hpp>
#include <Weight.hpp>
#include <Bias.hpp>
#include <Sigmoid.hpp>
#include <MUL.hpp>
#include <MSE.hpp>
#include <Utils.hpp>

//#include "spdlog/spdlog.h"
//using Eigen::MatrixXd;
int main() {
	auto graph = std::make_unique<Graph>();

	Eigen::MatrixXf mX1(4, 2);
	mX1 << 1, 0,
			0, 1,
			0, 0,
			1, 1;
	Eigen::MatrixXf C(4, 1);
	C << 1,
			1,
			0,
			0;
	auto CN = std::make_shared<Placeholder>(C);

	Eigen::MatrixXf mW1 = generateMatrix(0., 1., 2, 2);
	Eigen::MatrixXf b1 = generateMatrix(0., 10., 1, 2);

	Eigen::MatrixXf mW12 = generateMatrix(0., 1., 2, 2);

	Eigen::MatrixXf mW2 = generateMatrix(0., 1., 2, 1);
	Eigen::MatrixXf b2 = generateMatrix(0., 10., 1, 1);

//create Inputs, Weights and Biases
	auto X = std::make_shared<Placeholder>(mX1);
	auto W = std::make_shared<Weight>(mW1);
	auto B1 = std::make_shared<Bias>(b1);
	auto B2 = std::make_shared<Bias>(b2);
	auto W2 = std::make_shared<Weight>(mW2);
//create First hidden Layer
	auto mul = std::make_shared<MUL>(X, W);
	auto sum = std::make_shared<SUM>(mul, B1);
	auto sig1 = std::make_shared<Sigmoid>(sum);
	//create output layer
	auto mul2 = std::make_shared<MUL>(sig1, W2);
	auto sum2 = std::make_shared<SUM>(mul2, B2);
	auto sig2 = std::make_shared<Sigmoid>(sum2);
	//create Loss function
	auto mse = std::make_shared<MSE>(sig2, CN);

	Session session(mse, std::move(graph));
	session.run();
	std::cout << "First Run" << std::endl;
	std::cout << "Output:\n" << sig2->getForward() << std::endl;
	std::cout << "LOSS:\n" << mse->getForward() << std::endl;
	for (int i = 0; i < 5000; i++) {
		session.run();
	}
	session.run();
	std::cout << " Results of Last Run (5002th)" << std::endl;
	std::cout << "Output:\n" << sig2->getForward() << std::endl;
	std::cout << "LOSS:\n" << mse->getForward() << std::endl;
}