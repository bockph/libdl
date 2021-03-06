//
// Created by pbo on 24.07.19.
//
#pragma once

#include <Eigen/Dense>
#include <utility>
#include <utility>
/*!
 * Throughout the project only dynamic float Eigen matrices should be used
 */
using Matrix = Eigen::MatrixXf;
/*!
 * Defines possible Optimizer
 */
enum Optimizer {
	Adam
};
/*!
 * Defines possible Activation functions
 */
enum ActivationType {
	ReLu, Sigmoid,  None
};
/*!
 * Defines possible Initialization Types
 */
enum InitializationType {
	Xavier
};
/*!
 * Defines possible LossTypes
 */
enum LossType {
	CrossEntropy, MSE
};

/*!
 * This struct holds all kind of possible Hyper parameters, that may be used to update an Parameter during training
 */
struct HyperParameters {
	explicit HyperParameters(int epochs = 10, int batchSize = 8, float learningRate = 0.01,
							 Optimizer optimizer = Optimizer::Adam, float beta1 = 0.9, float beta2 = 0.999)
			: _epochs(epochs)
			, _batchSize(batchSize)
			, _learningRate(learningRate)
			, _optimizer(optimizer)
			, _beta1(beta1)
			, _beta2(beta2) {}

	~HyperParameters() = default;

	float _learningRate;
	float _beta1;
	float _beta2;
	int _batchSize;
	int _epochs;
	Optimizer _optimizer;

	std::string toString() {
		std::stringstream ss;
		ss << "Epochs: " << _epochs << std::endl;
		ss << "BatchSize: " << _batchSize << std::endl;
		ss << "Learning Rate: " << _learningRate << std::endl;
		ss << "Beta1: " << _beta1 << std::endl;
		ss << "Beta2: " << _beta2 << std::endl;
		return ss.str();
	}
};
/*!
 * This struct holds four vectors.
 * The first two holding the data and labels of the training set
 * The second two hild the data and labels of the validation set
 */
struct DataSet {
	DataSet() = default;;

	~DataSet() = default;

	DataSet(std::vector<Matrix> trainingSamples, std::vector<Matrix> trainingLabels)
			: _trainingSamples(std::move(std::move(trainingSamples))), _trainingLabels(std::move(std::move(trainingLabels))) {};

	DataSet(std::vector<Matrix> trainingSamples, std::vector<Matrix> trainingLabels, std::vector<Matrix>
	validationSamples, std::vector<Matrix> validationLabels)
			: _trainingSamples(std::move(trainingSamples))
			, _trainingLabels(std::move(trainingLabels))
			, _validationSamples(std::move(validationSamples))
			, _validationLabels(std::move(validationLabels)) {};

	std::vector<Matrix> _trainingSamples{}, _trainingLabels{};
	std::vector<Matrix> _validationSamples{}, _validationLabels{};
};

/*!
 * This struct holds the results of a training, storing in each field of a vecotr, the value for one epoch
 */
struct TrainingEvaluation {
	explicit TrainingEvaluation(HyperParameters hyperParameters)
			: _hyperParameters(hyperParameters) {}

	TrainingEvaluation(std::vector<float> trainingLoss, std::vector<float> trainingAccuracy,
					   HyperParameters hyperParameters)
			: _trainingLoss(std::move(trainingLoss))
			, _trainingAccuracy(std::move(trainingAccuracy))
			, _hyperParameters(hyperParameters) {}

	TrainingEvaluation(std::vector<float> trainingLoss, std::vector<float> trainingAccuracy,
					   std::vector<float> validationLoss, std::vector<float> validationAccuracy,
					   HyperParameters hyperParameters)
			: _trainingLoss(std::move(trainingLoss))
			, _trainingAccuracy(std::move(trainingAccuracy))
			, _validationLoss(std::move(validationLoss))
			, _validationAccuracy(std::move(validationAccuracy))
			, _hyperParameters(hyperParameters) {}

	std::vector<float> _trainingLoss;
	std::vector<float> _validationLoss;
	std::vector<float> _trainingAccuracy;
	std::vector<float> _validationAccuracy;



	HyperParameters _hyperParameters;
};