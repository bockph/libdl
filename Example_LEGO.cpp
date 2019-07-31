//
// Created by pbo on 18.06.19.
//

#include <iostream>
#include <Placeholder.hpp>
#include <Operation.hpp>
#include <SummationOp.hpp>
#include <MultiplicationOp.hpp>
#include <DataInitialization.hpp>
#include <Parameter.hpp>
#include <ConvolutionOp.hpp>
#include <ReLuOp.hpp>
#include <MaxPoolOp.hpp>
#include <SoftmaxOp.hpp>
#include <CrossEntropyOp.hpp>
#include "mnist/mnist_reader.hpp"
#include <mnist/mnist_utils.hpp>

#include <lodepng.hpp>
#include <iomanip>


#include <IO.hpp>
#include <InputLayer.hpp>
#include <ConvolutionLayer.hpp>
#include <AbstractLayer.hpp>
#include <MaxPoolLayer.hpp>
#include <DenseLayer.hpp>
#include <LossLayer.hpp>
#include <LogitsLayer.hpp>
#include <NeuralNetwork.hpp>
#include <filesystem>
#include <algorithm>
#include <random>
#include <LegoDataLoader.hpp>





int main() {
//
	/*
	 * batch_size: if this is changed '#define BATCH_SIZE' in Node.hpp has to be changed as well
	 * epochs: sets the amount of epochs for training, to big values in combination with a big 'amount_batches' can lead to OutOfMemory Error
	 * amount_batches: 'batch_size*amount_batches' gives the total amount of samples
	 * trainModel: defines if the model should be trained with the aboved set parameters
	 * testModel: defines if the model should be tested (using the MNIST test_data)
	 * writeWeights: if set the trained Weights  are written to Source_Directory/WeightDeposit
	 * readWeights: if set (and Weights have already been Written once) weights are initialized with weights from Source_Directory/WeightDeposit
	 */
	int batch_size = 16;
	int epochs = 5;
	double learningRate = 0.0001;
	int amount_batches = 10;
	bool trainModel = true;
	bool testModel = false;
	bool writeWeights = true;
	bool readWeights = false;

	std::vector<std::pair<std::string, int>> shuffledDataStrings = LegoDataLoader::shuffleData(DATA_DIR);
	DataSet legoData;
	LegoDataLoader::getData(batch_size * amount_batches, shuffledDataStrings, legoData);



/*
 * Create Neural Network
 */
	HyperParameters config(epochs, batch_size, learningRate);

	std::shared_ptr<Graph> graph = std::make_shared<Graph>(config);
	//Create InputLayer
	auto inputLayer = std::make_shared<InputLayer>(graph, batch_size, 160000, 4);

	//Convolutional Layer 1

	auto convolution1 = std::make_shared<ConvolutionLayer>(inputLayer, graph, ActivationType::ReLu,
			32, 8, 2, InitializationType::Xavier);

	auto maxPool1 = std::make_shared<MaxPoolLayer>(convolution1, graph, 2, 2);


	//convolutional Layer 2
	auto convolution2 = std::make_shared<ConvolutionLayer>(maxPool1, graph, ActivationType::ReLu, 64, 5, 2, InitializationType::Xavier);
	//Maxpooling
	auto maxPool2 = std::make_shared<MaxPoolLayer>(convolution2,graph, 2, 2);

	//Dense Layer 1
	auto dense1 = std::make_shared<DenseLayer>(maxPool2, graph, ActivationType::ReLu, 1024, InitializationType::Xavier);

	//Dense Layer 2
	auto dense2 = std::make_shared<DenseLayer>(dense1, graph, ActivationType::None, 16, InitializationType::Xavier);


	//Logits Layer
	auto logits = std::make_shared<LogitsLayer>(dense2, graph, 16);

	//    Cost Layer
	auto loss = std::make_shared<LossLayer>(logits, graph, LossType::CrossEntropy);

	//Create Deep Learning session
	NeuralNetwork network(graph, inputLayer, loss, config);

	/*
	 * Initialize Network with precalculated Weights
	 */
	if (readWeights) { network.readParameters(WEIGHT_DEPOSIT, "lego_layer"); }

	/*
	 * Train the Network
	 */
	if (trainModel) {
    std::cout<<"GO"<<std::endl;
		TrainingEvaluation eval =network.trainAndValidate(legoData, config);
//		network.train(legoData,config);
		/*
		 * Write calculated Weights to Network
		 */
		if (writeWeights) { network.writeParameters(WEIGHT_DEPOSIT, "lego_layer"); }
	}



	/*
	 * Test the Network
	 */

	if (testModel) {

		/*for (int b = 0; b < amount_batches; b++) {

			network.run(test_data[b], test_label[b]);

			Eigen::MatrixXf::Index maxRow, maxCol;

			for (int i = 0; i < test_data[b].rows(); i++) {
				logits->getOutputNode()->getForward().block(i, 0, 1, 10).maxCoeff(&maxRow, &maxCol);
				int p = maxCol;
				test_label[b].block(i, 0, 1, 10).maxCoeff(&maxRow, &maxCol);
				int A = maxCol;

				if (p == A) { correct++; }
				total++;

			}


		}
		std::cout << "Amount Correct: " << correct << "\nAmount Wrong: " << total - correct << "\nPercentage Correct: "
				  << correct / (float) total << std::endl;*/
	}


}