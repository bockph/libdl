//
// Created by pbo on 24.07.19.
//
#pragma once

enum Optimizer {
	Adam
};

struct hyperParameters {
	hyperParameters(int epochs = 10, int batchsize = 8, float learningRate = 0.01, Optimizer optimizer =
	Optimizer::Adam,
					float beta1 = 0.9, float beta2 = 0.999)
			:
			_epochs(epochs)
			, _batchsize(batchsize)
			, _learningRate(learningRate)
			, _optimizer(optimizer)
			, _beta1(beta1)
			, _beta2(beta2) {}

	float _learningRate;
	float _beta1;
	float _beta2;
	int _batchsize;
	int _epochs;
	Optimizer _optimizer;
};

struct dataSet {
	dataSet(){};
	dataSet(std::vector<Matrix> trainingSamples, std::vector<Matrix> trainingLabels)
			: _trainingSamples(trainingSamples), _trainingLabels(trainingLabels) {};

	dataSet(std::vector<Matrix> trainingSamples, std::vector<Matrix> trainingLabels, std::vector<Matrix>
	validationSamples, std::vector<Matrix> validationLabels)
			: _trainingSamples(trainingSamples)
			, _trainingLabels(trainingLabels)
			, _validationSamples(validationSamples)
			, _validationLabels(validationLabels) {};

	std::vector<Matrix> _trainingSamples, _trainingLabels;
	std::vector<Matrix> _validationSamples, _validationLabels;
};