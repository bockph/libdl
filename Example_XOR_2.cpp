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
#include <Softmax.hpp>
#include <CrossEntropyLoss.hpp>


int main() {
	auto graph = std::make_unique<Graph>();
	//Input Data: with dimensions: Amount of Training Samples x Dimension of training Sample
	Eigen::MatrixXf mX1(4, 2);
	mX1 << 1, 0,
			0, 1,
			0, 0,
			1, 1;
	//Corresponding Classes of input Data DImensions are : Amount of Training Samples x 1
	Eigen::MatrixXf C(4, 2);
	C << 	1,0,
			1,0,
			0,1,
			0,1;
	auto CN = std::make_shared<Placeholder>(C,0,0);
	//Weights Hidden Layer 1
	Eigen::MatrixXf mW1 = generateRandomMatrix(0., 1., 2, 2);
	//Bias Hidden Layer 1
	Eigen::MatrixXf b1 = generateRandomMatrix(0., 0., 1, 2);

	//Weights OutputLayer
	Eigen::MatrixXf mW2 = generateRandomMatrix(0., 1., 2, 2);
	//Bias Output Layer
	Eigen::MatrixXf b2 = generateRandomMatrix(0., 0., 1, 2);

	//create Inputs, Weights and Biases
	auto X = std::make_shared<Placeholder>(mX1,0,0);
	auto W = std::make_shared<Weight>(mW1);
	auto W2 = std::make_shared<Weight>(mW2);
	auto B1 = std::make_shared<Bias>(b1,0);
	auto B2 = std::make_shared<Bias>(b2,0);

	//create First hidden Layer
	auto mul = std::make_shared<MUL>(X, W);
	auto sum = std::make_shared<SUM>(mul, B1);
	auto sig1 = std::make_shared<Sigmoid>(sum);

	//create output layer
	auto mul2 = std::make_shared<MUL>(sig1, W2);
	auto sum2 = std::make_shared<SUM>(mul2, B2);
//	auto sig2 = std::make_shared<Sigmoid>(sum2);
	auto soft = std::make_shared<Softmax>(sum2,2);

	//create Loss function
	auto mse = std::make_shared<CrossEntropyLoss>(soft, CN);

	//Create Deep Learning session
	Session session(mse, std::move(graph));

	//session.run() Executes Forward Pass & Backpropagation, Learning Rate is hardcoded at the moment and is 1
	session.run();
	std::cout << "First Run" << std::endl;
	std::cout << "Output:\n" << soft->getForward() << std::endl;
	std::cout << "LOSS:\n" << mse->getForward() << std::endl;
	for (int i = 0; i < 5000000; i++) {
		session.run();
	}
	session.run();
	std::cout << " Results of Last Run (5002th)" << std::endl;
	std::cout << "Output:\n" << soft->getForward() << std::endl;
	std::cout << "LOSS:\n" << mse->getForward() << std::endl;
}