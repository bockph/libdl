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

/*void getBatches(int batch_size, int amountBatches, std::vector<Eigen::MatrixXf> &training_data,
                std::vector<Eigen::MatrixXf> &label_data, bool trainData = true) {

    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION,
                                                                            batch_size * amountBatches);
    mnist::normalize_dataset(dataset);

    training_data.clear();
    label_data.clear();

    for (int j = 0; j < amountBatches; j++) {
        Eigen::MatrixXf img(batch_size, 28 * 28);
        for (int i = 0; i < batch_size; i++) {
            if (trainData) {
                Eigen::Matrix<unsigned char, 1, 784> tmp(dataset.training_images.at(i + j * batch_size).data());
                Eigen::MatrixXf tmp2 = tmp.cast<float>();
                img.block(i, 0, 1, 784) = tmp2;
            } else {
                if (i + j * batch_size < 5000) {
                    Eigen::Matrix<unsigned char, 1, 784> tmp(dataset.test_images.at(i + j * batch_size).data());
                    Eigen::MatrixXf tmp2 = tmp.cast<float>();
                    img.block(i, 0, 1, 784) = tmp2;
                }

            }

        }


        training_data.push_back(img);
        Eigen::MatrixXf C(batch_size, 10);
        C.setZero();
        for (int i = 0; i < batch_size; i++) {
            if (trainData)
                C(i, dataset.training_labels.at(i + j * batch_size)) = 1;
            else {
                if (i + j * batch_size < 5000) {
                    C(i, dataset.test_labels.at(i + j * batch_size)) = 1;
                }
            }


        }

        label_data.push_back(C);
    }


}*/

void getData(int samples, DataSet &data) {

	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
			mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION, samples);
	mnist::normalize_dataset(dataset);
	std::vector<Matrix> trainingSamples, trainingLabels, testSamples, testLabels;

	for (int i = 0; i < samples; i++) {
		Eigen::Matrix<unsigned char, 1, 784> train(dataset.training_images.at(i).data());
		Eigen::MatrixXf labelTrain(1, 10);
		labelTrain.setZero();
		labelTrain(0, dataset.training_labels[i]) = 1;

		trainingSamples.push_back(train.cast<float>());
		trainingLabels.push_back(labelTrain);

		if (i < 5000) {
			Eigen::Matrix<unsigned char, 1, 784> test(dataset.test_images.at(i).data());
			testSamples.push_back(test.cast<float>());
			Eigen::MatrixXf labelTest(1, 10);
			labelTest.setZero();
			labelTest(0, dataset.training_labels[i]) = 1;
			testLabels.push_back(labelTest);

		}


	}
	data._trainingLabels = trainingLabels;
	data._trainingSamples = trainingSamples;
	data._validationLabels = testSamples;
	data._validationSamples = testLabels;


}


int main() {
	Eigen::initParallel();
	/*
	 * batch_size: if this is changed '#define BATCH_SIZE' in Node.hpp has to be changed as well
	 * epochs: sets the amount of epochs for training, to big values in combination with a big 'amount_batches' can lead to OutOfMemory Error
	 * amount_batches: 'batch_size*amount_batches' gives the total amount of samples
	 * trainModel: defines if the model should be trained with the aboved set parameters
	 * testModel: defines if the model should be tested (using the MNIST test_data)
	 * writeWeights: if set the trained Weights  are written to Source_Directory/data/STORAGE
	 * readWeights: if set (and Weights have already been Written once) weights are initialized with weights from Source_Directory/WeightDeposit
	 */
	int batch_size = 32;
	int epochs = 50;
	int amount_batches = 10;
	double learningRate = 0.001;
	bool trainModel = true;
	bool testModel = true;
	bool writeWeights = false;
	bool readWeights = false;


	std::vector<Eigen::MatrixXf> training_data, training_label;
	std::vector<Eigen::MatrixXf> test_data, test_label;


	DataSet data;
	getData(batch_size * amount_batches, data);



/*
 * Create Neural Network
 */

	HyperParameters config(epochs, batch_size, learningRate);
	std::shared_ptr<Graph> graph = std::make_shared<Graph>();


	//Create InputLayer
	auto inputLayer = std::make_shared<InputLayer>(graph, batch_size, 28*28, 1);

	//Convolutional Layer 1

	auto convolution1 = std::make_shared<ConvolutionLayer>(inputLayer, graph, ActivationType::ReLu, 32, 5, 1, InitializationType::Xavier);

	auto maxPool2 = std::make_shared<MaxPoolLayer>(convolution1, graph, 2, 2);


	//convolutional Layer 2
	auto convolution2 = std::make_shared<ConvolutionLayer>(maxPool2, graph, ActivationType::ReLu, 64, 5, 1, InitializationType::Xavier);
	//Maxpooling
	auto maxPool = std::make_shared<MaxPoolLayer>(convolution2, graph, 2, 2);

	//Dense Layer 1
	auto dense1 = std::make_shared<DenseLayer>(maxPool, graph, ActivationType::ReLu, 1024, InitializationType::Xavier);

	//Dense Layer 2
	auto dense2 = std::make_shared<DenseLayer>(dense1, graph, ActivationType::None, 10, InitializationType::Xavier);


	//Logits Layer
	auto logits = std::make_shared<LogitsLayer>(dense2, graph, 10);

	//    Cost Layer
	auto loss = std::make_shared<LossLayer>(logits, graph, LossType::CrossEntropy);

	//Create Deep Learning session
	NeuralNetwork network(graph, inputLayer, loss);

	/*
	 * Initialize Network with precalculated Weights
	 */
	if (readWeights) { network.readParameters(STORAGE, "mnist_layer"); }

	/*
	 * Train the Network
	 */
	if (trainModel) {
		//trainAndValidate causes right now some Problems check Data preparation
		network.train(data, config);

		/*
		 * Write calculated Weights to Network
		 */
		if (writeWeights) { network.writeParameters(STORAGE, "mnist_layer"); }
	}





}