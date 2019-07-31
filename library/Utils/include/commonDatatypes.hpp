//
// Created by pbo on 24.07.19.
//
#pragma once

#include <Eigen/Dense>

using Matrix = Eigen::MatrixXf;

enum Optimizer {
	Adam
};
enum ActivationType {
	ReLu, Sigmoid, LeakyReLu, None
};
enum InitializationType {
	Xavier
};
enum LossType {
	CrossEntropy, MSE
};

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

struct DataSet {
	DataSet() {};

	~DataSet() = default;

	DataSet(std::vector<Matrix> trainingSamples, std::vector<Matrix> trainingLabels)
			: _trainingSamples(trainingSamples), _trainingLabels(trainingLabels) {};

	DataSet(std::vector<Matrix> trainingSamples, std::vector<Matrix> trainingLabels, std::vector<Matrix>
	validationSamples, std::vector<Matrix> validationLabels)
			: _trainingSamples(trainingSamples)
			, _trainingLabels(trainingLabels)
			, _validationSamples(validationSamples)
			, _validationLabels(validationLabels) {};

	std::vector<Matrix> _trainingSamples{}, _trainingLabels{};
	std::vector<Matrix> _validationSamples{}, _validationLabels{};
};

struct TrainingEvaluation {
	explicit TrainingEvaluation(HyperParameters hyperParameters)
			: _hyperParameters(hyperParameters) {}

	TrainingEvaluation(std::vector<float> trainingLoss, std::vector<float> trainingAccuracy,
					   HyperParameters hyperParameters)
			: _trainingLoss(trainingLoss)
			, _trainingAccuracy(trainingAccuracy)
			, _hyperParameters(hyperParameters) {}

	TrainingEvaluation(std::vector<float> trainingLoss, std::vector<float> trainingAccuracy,
					   std::vector<float> validationLoss, std::vector<float> validationAccuracy,
					   HyperParameters hyperParameters)
			: _trainingLoss(trainingLoss)
			, _trainingAccuracy(trainingAccuracy)
			, _validationLoss(validationLoss)
			, _validationAccuracy(validationAccuracy)
			, _hyperParameters(hyperParameters) {}

	std::vector<float> _trainingLoss;
	std::vector<float> _validationLoss;
	std::vector<float> _trainingAccuracy;
	std::vector<float> _validationAccuracy;

	const std::vector<float> &getTrainingLoss() const {
		return _trainingLoss;
	}

	const std::vector<float> &getValidationLoss() const {
		return _validationLoss;
	}

	const std::vector<float> &getTrainingAccuracy() const {
		return _trainingAccuracy;
	}

	const std::vector<float> &getValidationAccuracy() const {
		return _validationAccuracy;
	}

	const HyperParameters &getHyperParameters() const {
		return _hyperParameters;
	}

	HyperParameters _hyperParameters;
};