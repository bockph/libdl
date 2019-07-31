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
                  const std::shared_ptr<LossLayer> lossLayer);

    void trainBatch(Matrix &miniBatch, Matrix &labels, HyperParameters& params);

    Matrix predictBatch(Matrix &miniBatch, Matrix &labels);

    bool writeParameters(std::string dir, std::string networkName);

    void testAccuracy(Matrix &results, Matrix &labels, float &correct, float &total);


    bool writeTrainingLog(std::string dir, std::string training);

    bool readParameters(std::string dir, std::string networkName);


    float getLoss();

    float train(DataSet &data, HyperParameters params, float trainingLossThreshold = 1);

    TrainingEvaluation trainAndValidate(DataSet &data, HyperParameters params, float trainingLossThreshold = 1);

	static std::vector<Matrix> extractBatchList(std::vector<Matrix> &dataset, int batchSize);

private:


    std::shared_ptr<Graph> _computeGraph;
    std::shared_ptr<InputLayer> _inputLayer;
    std::shared_ptr<LossLayer> _lossLayer;
    bool _runAchieved;

};


