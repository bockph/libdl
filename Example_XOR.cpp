#include <iostream>
//#include <Eigen/Dense>
#include <Placeholder.hpp>
#include <Operation.hpp>
#include <SummationOp.hpp>
#include <Parameter.hpp>
#include <SigmoidOP.hpp>
#include <MultiplicationOp.hpp>
#include <MSEOp.hpp>
#include <DataInitialization.hpp>
#include <Graph.hpp>
#include <NeuralNetwork.hpp>
#include <LogitsLayer.hpp>
#include <InputLayer.hpp>
#include <DenseLayer.hpp>


int main() {
	int epochs = 10;
	int batchSize = 4;
	int learningRate = 10;

	HyperParameters config(epochs, batchSize, learningRate);

	std::shared_ptr<Graph> graph = std::make_shared<Graph>();
	/*
	 * Setup Layers
	 */
	auto inputLayer = std::make_shared<InputLayer>(graph, batchSize, 2, 1);
	auto dense1 = std::make_shared<DenseLayer>(inputLayer, graph, ActivationType::ReLu, 2,
			InitializationType::Xavier);
	auto dense2 = std::make_shared<DenseLayer>(dense1, graph, ActivationType::ReLu, 2,
			InitializationType::Xavier);
	auto softmax = std::make_shared<LogitsLayer>(dense2, graph, 2);
	auto loss = std::make_shared<LossLayer>(softmax, graph, LossType::CrossEntropy);

	NeuralNetwork network(graph, inputLayer, loss);
	/*
	 * Prepare Data Inputs
	 */
	Matrix input1(1, 2);
	input1 << 1, 0;
	Matrix input2(1, 2);
	input2 << 0, 1;
	Matrix input3(1, 2);
	input3 << 0, 0;
	Matrix input4(1, 2);
	input4 << 1, 1;
	/*
	 * Prepare corresponding Labels to Data Inputs
	 */
	Matrix label1(1, 2);
	label1 << 1, 0;
	Matrix label2(1, 2);
	label2 << 1, 0;

	Matrix label3(1, 2);
	label3 << 0, 1;
	Matrix label4(1, 2);
	label4 << 0, 1;

	/*
	 * Create Data set
	 */
	std::vector<Matrix> trainingData{input1, input2, input3, input4}, trainingLabels{label1, label2, label3, label4};
	DataSet data(trainingData, trainingLabels);

	/*
	 * Train Network
	 */
	network.train(data, config);
	std::cout << "Prediction:\n" << softmax->getOutputNode()->getForward() << std::endl;

	std::cout << "Final Loss:\n" << loss->getOutputNode()->getForward() << std::endl;


}