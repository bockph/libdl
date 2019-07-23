#include <iostream>
//#include <Eigen/Dense>
#include <Placeholder.hpp>
#include <Operation.hpp>
#include <SummationOp.hpp>
#include <Session.hpp>
#include <Weight.hpp>
#include <Bias.hpp>
#include <SigmoidOP.hpp>
#include <MultiplicationOp.hpp>
#include <MSEOp.hpp>
#include <DataInitialization.hpp>

int main() {
	//Input Data: with dimensions: Amount of Training Samples x Dimension of training Sample
	Eigen::MatrixXf mX1(4, 2);
	mX1 << 1, 0,
			0, 1,
			0, 0,
			1, 1;
	//Corresponding Classes of input Data DImensions are : Amount of Training Samples x 1
	Eigen::MatrixXf C(4, 1);
	C << 1,
			1,
			0,
			0;

	auto CN = std::make_shared<Placeholder>(C,0,0);
	//Weights Hidden Layer 1
	Eigen::MatrixXf mW1 = DataInitialization::generateRandomMatrix(0., 1., 2, 2);
	//Bias Hidden Layer 1
	Eigen::MatrixXf b1 = DataInitialization::generateRandomMatrix(0., 10., 1, 2);

	//Weights OutputLayer
	Eigen::MatrixXf mW2 = DataInitialization::generateRandomMatrix(0., 1., 2, 1);
	//Bias Output Layer
	Eigen::MatrixXf b2 = DataInitialization::generateRandomMatrix(0., 10., 1, 1);

	//create Inputs, Weights and Biases
	auto X = std::make_shared<Placeholder>(mX1,0,0);
	auto W = std::make_shared<Weight>(mW1);
	auto W2 = std::make_shared<Weight>(mW2);
	auto B1 = std::make_shared<Bias>(b1,0);
	auto B2 = std::make_shared<Bias>(b2,0);

	//create First hidden Layer
	auto mul = std::make_shared<MultiplicationOp>(X, W);
	auto sum = std::make_shared<SummationOp>(mul, B1);
	auto sig1 = std::make_shared<SigmoidOP>(sum);

	//create output layer
	auto mul2 = std::make_shared<MultiplicationOp>(sig1, W2);
	auto sum2 = std::make_shared<SummationOp>(mul2, B2);
	auto sig2 = std::make_shared<SigmoidOP>(sum2);

	//create Loss function
	auto mse = std::make_shared<MSEOp>(sig2, CN);

	//Create Deep Learning session
	Session session(mse);

	//session.run() Executes Forward Pass & Backpropagation, Learning Rate is hardcoded at the moment and is 1
	session.run();
	std::cout << "First Run" << std::endl;
	std::cout << "Output:\n" << sig2->getForward() << std::endl;
	std::cout << "LOSS:\n" << mse->getForward() << std::endl;
	for (int i = 0; i < 50000; i++) {
		session.run();
	}
	session.run();
	std::cout << " Results of Last Run (5002th)" << std::endl;
	std::cout << "Output:\n" << sig2->getForward() << std::endl;
	std::cout << "LOSS:\n" << mse->getForward() << std::endl;
}