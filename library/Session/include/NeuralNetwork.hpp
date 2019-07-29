//
// Created by pbo on 24.07.19.
//

#pragma once

#include <InputLayer.hpp>
#include <LossLayer.hpp>
#include <commonDatatypes.hpp>
#include <Graph.hpp>

class NeuralNetwork {
public:
	NeuralNetwork(const std::shared_ptr<Graph> computeGraph, const std::shared_ptr<InputLayer> inputLayer,
				  const std::shared_ptr<LossLayer> lossLayer, const hyperParameters params = hyperParameters());

	void trainBatch(Matrix &miniBatch, Matrix &labels);

	void predictBatch(Matrix &miniBatch, Matrix &labels);

	bool writeVariables(std::string dir, std::string networkName);

	bool readVariables(std::string dir, std::string networkName);

	const hyperParameters &getParams() const;

	void setParams(const hyperParameters &params);

	float getLoss();

	float train(dataSet &data, hyperParameters params,float trainingLossThreshold=1);

	std::vector<std::pair<float, float>>
	trainAndValidate(dataSet &data, hyperParameters params,float trainingLossThreshold=1);

	std::vector<Matrix> extractBatchList(std::vector<Matrix> &dataset, int batchSize);

private:
	std::shared_ptr<Graph> _computeGraph;
	std::shared_ptr<InputLayer> _inputLayer;
	std::shared_ptr<LossLayer> _lossLayer;
	bool _runAchieved;

};


