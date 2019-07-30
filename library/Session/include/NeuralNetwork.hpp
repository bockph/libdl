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
				  const std::shared_ptr<LossLayer> lossLayer, const HyperParameters params = HyperParameters());

	void trainBatch(Matrix &miniBatch, Matrix &labels);

	void predictBatch(Matrix &miniBatch, Matrix &labels);

	bool writeVariables(std::string dir, std::string networkName);

	bool readVariables(std::string dir, std::string networkName);

	void setParams(const HyperParameters &params);

	float getLoss();

	float train(DataSet &data, HyperParameters params,float trainingLossThreshold=1);

	std::vector<std::pair<float, float>>
	trainAndValidate(DataSet &data, HyperParameters params,float trainingLossThreshold=1);


private:
	std::vector<Matrix> extractBatchList(std::vector<Matrix> &dataset, int batchSize);


	std::shared_ptr<Graph> _computeGraph;
	std::shared_ptr<InputLayer> _inputLayer;
	std::shared_ptr<LossLayer> _lossLayer;
	bool _runAchieved;

};


