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
#include <SigmoidOP.hpp>
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

void getBatches(int batch_size, int amountBatches, std::vector<Eigen::MatrixXf> &training_data,
				std::vector<Eigen::MatrixXf> &label_data, bool trainData = true) {


	std::vector<std::vector<unsigned char>> data;


	unsigned width, height;
	int amountFiles = amountBatches * batch_size;
	int counter = 0;
	for (int i = 1;; i++) {
		for (int s = 1; s < 17; s++) {
			if (counter >= amountFiles) { goto end; }
			std::stringstream stream;
			stream << DATA_DIR << "lego/train/" << s << "/" << std::setw(4) << std::setfill('0') << i << ".png";
			//read png and store to vector
			std::vector<unsigned char> image;
			unsigned error = lodepng::decode(image, width, height, stream.str());
			if (error) { std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl; }

			data.push_back(image);
			counter++;

		}
	}
	end:

	int totalSize = width * height;
	training_data.clear();
	label_data.clear();
	counter = 0;
	for (int j = 0; j < amountBatches; j++) {
		Eigen::MatrixXf img(batch_size, width * height * 4);
		Eigen::MatrixXf C(batch_size, 16);
		C.setZero();
		for (int i = 0; i < batch_size; i++) {
			if (trainData) {
				Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> tmp(1, 160000);
				int idx = 0;
				for (int k = 0; k < data[i + j * batch_size].size() - 4; k += 4) {
					tmp(0, idx) = data[i + j * batch_size][k];
					tmp(0, idx + totalSize) = data[i + j * batch_size][k + 1];
					tmp(0, idx + totalSize * 2) = data[i + j * batch_size][k + 2];
					tmp(0, idx + totalSize * 3) = data[i + j * batch_size][k + 3];
					idx++;
				}
//				.data());
				Eigen::MatrixXf tmp2 = tmp.cast<float>();
				img.block(i, 0, 1, 160000) = tmp2;

				C(i, counter) = 1;
//				std::cout<<"Counter: "<<counter<<"   :"<<C.row(i)<<std::endl;
				counter++;
				if (counter == 16) { counter = 0; }

			} else {
//                if (i + j * batch_size < 5000) {
//                    Eigen::Matrix<unsigned char, 1, 784> tmp(dataset.test_images.at(i + j * batch_size).data());
//                    Eigen::MatrixXf tmp2 = tmp.cast<float>();
//                    img.block(i, 0, 1, 784) = tmp2;
//                }

			}


		}

		img /= 255;
		training_data.push_back(img);
		/*Eigen::MatrixXf C(batch_size, 10);
		C.setZero();
		for (int i = 0; i < batch_size; i++) {
			if (trainData) {
//				C(i, counter) = 1;
//				if (counter == 16) { counter = 0; }
//				counter++;
			} else {
				if (i + j * batch_size < 5000) {
					C(i, dataset.test_labels.at(i + j * batch_size)) = 1;
				}
			}


		}*/
		label_data.push_back(C);
	}


}


void getData(int samples, std::vector<std::pair<std::string, int>> shuffledFiles, dataSet &data) {
	int numberValidation = std::floor(samples * 0.2);
	assert(samples + numberValidation <= shuffledFiles.size());

	std::vector<Matrix> trainingSamples, trainingLabels;
	std::vector<Matrix> validationSamples, validationLabels;

	unsigned width, height;

	for (int i = 0; i < samples; i++) {
		std::vector<unsigned char> image;
		unsigned error = lodepng::decode(image, width, height, shuffledFiles[i].first);
		if (error) { std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl; }
		int totalSize = width * height;

		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> tmp(1, 160000);
		int idx = 0;
		for (int k = 0; k < shuffledFiles[i].first.size() - 4; k += 4) {
			tmp(0, idx) = shuffledFiles[i].first[k];
			tmp(0, idx + totalSize) = shuffledFiles[i].first[k + 1];
			tmp(0, idx + totalSize * 2) = shuffledFiles[i].first[k + 2];
			tmp(0, idx + totalSize * 3) = shuffledFiles[i].first[k + 3];
			idx++;
		}
		Matrix sample = tmp.cast<float>();
		sample /= 255;
		Matrix C(1, 16);
		C.setZero();
		C(0, shuffledFiles[i].second) = 1;
		trainingSamples.push_back(sample);
		trainingLabels.push_back(C);


	}
	for (int i = samples; i < samples + numberValidation; i++) {
		std::vector<unsigned char> image;
		unsigned error = lodepng::decode(image, width, height, shuffledFiles[i].first);
		if (error) { std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl; }
		int totalSize = width * height;

		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> tmp(1, 160000);
		int idx = 0;
		for (int k = 0; k < shuffledFiles[i].first.size() - 4; k += 4) {
			tmp(0, idx) = shuffledFiles[i].first[k];
			tmp(0, idx + totalSize) = shuffledFiles[i].first[k + 1];
			tmp(0, idx + totalSize * 2) = shuffledFiles[i].first[k + 2];
			tmp(0, idx + totalSize * 3) = shuffledFiles[i].first[k + 3];
			idx++;
		}
		Matrix sample = tmp.cast<float>();
		sample /= 255;
		Matrix C(1, 16);
		C.setZero();
		C(0, shuffledFiles[i].second) = 1;
		validationSamples.push_back(sample);
		validationLabels.push_back(C);


	}

	data = {trainingSamples, trainingLabels, validationSamples, validationLabels};


}


std::vector<std::pair<std::string, int>> shuffleData() {
	std::ofstream(DATA_DIR + std::string("lego/shuffledData.csv"));
	std::vector<std::pair<std::string, int>> values;

	for (int i = 0; i < 16; i++) {
		std::string path = DATA_DIR + std::string("lego/train/") + std::to_string(i + 1) + std::string("/");
		for (const auto &entry : std::filesystem::directory_iterator(path)) {
			values.push_back(std::pair<std::string, int>(entry.path(), i));
		}// << std::endl;

	}
	auto rng = std::default_random_engine{};

	std::shuffle(std::begin(values), std::end(values), rng);

	return values;
}

int main() {
//	Eigen::setNbThreads(2);
	std::cout << Eigen::nbThreads() << std::endl;

//	Eigen::initParallel();
	/*
	 * batch_size: if this is changed '#define BATCH_SIZE' in Node.hpp has to be changed as well
	 * epochs: sets the amount of epochs for training, to big values in combination with a big 'amount_batches' can lead to OutOfMemory Error
	 * amount_batches: 'batch_size*amount_batches' gives the total amount of samples
	 * trainModel: defines if the model should be trained with the aboved set parameters
	 * testModel: defines if the model should be tested (using the MNIST test_data)
	 * writeWeights: if set the trained Weights  are written to Source_Directory/WeightDeposit
	 * readWeights: if set (and Weights have already been Written once) weights are initialized with weights from Source_Directory/WeightDeposit
	 */
	int batch_size = 4;
	int epochs = 5;
	double learningRate = 0.0001;
	int amount_batches = 10;
	bool trainModel = true;
	bool testModel = false;
	bool writeWeights = true;
	bool readWeights = false;

	std::vector<std::pair<std::string, int>> shuffledDataStrings = shuffleData();
	dataSet legoData;
	getData(batch_size * amount_batches, shuffledDataStrings, legoData);



/*
 * Create Neural Network
 */
	hyperParameters config(epochs, batch_size, learningRate);

	std::shared_ptr<Graph> graph = std::make_shared<Graph>(config);
	//Create InputLayer
	auto inputLayer = std::make_shared<InputLayer>(graph, batch_size, 200, 4);

	//Convolutional Layer 1

	auto convolution1 = std::make_shared<ConvolutionLayer>(inputLayer, graph, AbstractLayer::ActivationType::ReLu,
			32, 8, 2, AbstractLayer::InitializationType::Xavier);

	auto maxPool2 = std::make_shared<MaxPoolLayer>(convolution1, graph, 2, 2);


	//convolutional Layer 2
	auto convolution2 = std::make_shared<ConvolutionLayer>(maxPool2, graph, AbstractLayer::ActivationType::ReLu, 64, 5, 2, AbstractLayer::InitializationType::Xavier);
	//Maxpooling
	auto maxPool = std::make_shared<MaxPoolLayer>(convolution2,graph, 2, 2);

	//Dense Layer 1
	auto dense1 = std::make_shared<DenseLayer>(maxPool, graph, AbstractLayer::ActivationType::ReLu, 1024, AbstractLayer::InitializationType::Xavier);

	//Dense Layer 2
	auto dense2 = std::make_shared<DenseLayer>(dense1, graph, AbstractLayer::ActivationType::None, 16, AbstractLayer::InitializationType::Xavier);


	//Logits Layer
	auto logits = std::make_shared<LogitsLayer>(dense2, graph, 16);

	//    Cost Layer
	auto loss = std::make_shared<LossLayer>(logits, graph, AbstractLayer::LossType::CrossEntropy);

	//Create Deep Learning session
	NeuralNetwork network(graph, inputLayer, loss, config);

	/*
	 * Initialize Network with precalculated Weights
	 */
	if (readWeights) { network.readVariables(WEIGHT_DEPOSIT, "lego_layer"); }

	/*
	 * Train the Network
	 */
	if (trainModel) {

		network.trainAndValidate(legoData, config);
//		network.train(legoData,config);
		/*
		 * Write calculated Weights to Network
		 */
		if (writeWeights) { network.writeVariables(WEIGHT_DEPOSIT, "lego_layer"); }
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