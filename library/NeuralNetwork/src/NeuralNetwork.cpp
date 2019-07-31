//
// Created by pbo on 24.07.19.
//

#include <iostream>
#include "NeuralNetwork.hpp"
#include "../../Utils/include/commonDatatypes.hpp"

NeuralNetwork::NeuralNetwork(const std::shared_ptr<Graph> computeGraph, const std::shared_ptr<InputLayer> inputLayer,
                             const std::shared_ptr<LossLayer> lossLayer, const HyperParameters params)
        :
        _computeGraph(computeGraph), _inputLayer(inputLayer), _lossLayer(lossLayer), _runAchieved(false) {
}

void NeuralNetwork::trainBatch(Matrix &miniBatch, Matrix &labels,HyperParameters& params) {
    _inputLayer->updateX(miniBatch);
    _lossLayer->updateLabels(labels);

	_computeGraph->computeForward();
	_computeGraph->computeBackwards();
	_computeGraph->updateParameters(params);
    _runAchieved = true;

}

Matrix NeuralNetwork::predictBatch(Matrix &miniBatch, Matrix &labels) {
    _inputLayer->updateX(miniBatch);
    _lossLayer->updateLabels(labels);
	_computeGraph->computeForward();
    _runAchieved = true;
    return _lossLayer->getPrediction();
}

bool NeuralNetwork::writeVariables(std::string dir, std::string networkName) {
    return _computeGraph->writeVariables(dir + networkName);
}

bool NeuralNetwork::readVariables(std::string dir, std::string networkName) {
    return _computeGraph->readVariables(dir + networkName);

}




float NeuralNetwork::getLoss() {
    if (!_runAchieved) {
        throw std::runtime_error("The Neural Network can't return a Loss, if no runs has been performed yet.");
    }
    return _lossLayer->getLoss();
}

float NeuralNetwork::train(DataSet &data, HyperParameters params, float trainingLossThreshold) {
    if (data._trainingSamples.size() != data._trainingLabels.size()) {
        throw std::runtime_error("The size of the Samples does not equal the Size of the Labels");
    }
    std::vector<Matrix> sampleBatches = extractBatchList(data._trainingSamples, params._batchsize);
    std::vector<Matrix> labelBatches = extractBatchList(data._trainingLabels, params._batchsize);
    float cost = 0;
    for (int k = 0; k < params._epochs; k++) {
        cost = 0;
        //train Batch
        for (int i = 0; i < sampleBatches.size(); i++) {
            trainBatch(sampleBatches[i], labelBatches[i],params);
            cost += getLoss();
        }
        cost /= (float) sampleBatches.size();
        std::cout << "Current Cost:" << cost << " Round: " << k << std::endl;
        if (cost < trainingLossThreshold) { break; }

    }
    return cost;
}


TrainingEvaluation NeuralNetwork::trainAndValidate(DataSet &data, HyperParameters params, float trainingLossThreshold) {
    if (data._trainingSamples.size() != data._trainingLabels.size()) {
        throw std::runtime_error(
                "The size of the training Samples (" + std::to_string(data._trainingSamples.size()) +
                ")does not equal the Size of the training Labels (" + std::to_string(data._trainingLabels.size()) +
                ")");
    }
    if (data._validationSamples.size() != data._validationLabels.size()) {
        throw std::runtime_error(
                "The size of the validation Samples (" + std::to_string(data._validationSamples.size()) +
                ")does not equal the Size of the validation Labels (" + std::to_string(data._validationLabels.size()) +
                ")");
    }

    /*
     * Create Training Evaluation Object and calculation Variables
     */
    TrainingEvaluation evaluation(params);
    float trainingCost = 0, validationCost = 0;
    float correctTraining=0, correctValidation=0, totalTraining=0, totalValidation=0;

    /*
     * Extract Batches
     */
    std::vector<Matrix> sampleTrainBatches = extractBatchList(data._trainingSamples, params._batchsize);
    std::vector<Matrix> labelTrainBatches = extractBatchList(data._trainingLabels, params._batchsize);
    std::vector<Matrix> sampleValidationBatches = extractBatchList(data._validationSamples, params._batchsize);
    std::vector<Matrix> labelValidationBatches = extractBatchList(data._validationLabels, params._batchsize);


   /*
    * Start Training for Epochs=params._epochs
    */
    for (int k = 0; k < params._epochs; k++) {
        trainingCost = 0,validationCost = 0;
        correctTraining=0, correctValidation=0, totalTraining=0, totalValidation=0;
        //First Do Training
        for (int i = 0; i < sampleTrainBatches.size(); i++) {
            trainBatch(sampleTrainBatches[i], labelTrainBatches[i],params);
        }
        /*
         * Second evaluate Training and Validationset
         */
        for (int i = 0; i < sampleTrainBatches.size(); i++) {
            Matrix prediction = predictBatch(sampleTrainBatches[i], labelTrainBatches[i]);
            trainingCost += getLoss();
            testAccuracy(prediction, labelTrainBatches[i], correctTraining, totalTraining);
        }
        for (int i = 0; i < sampleValidationBatches.size(); i++) {
            Matrix prediction = predictBatch(sampleValidationBatches[i], labelValidationBatches[i]);
            validationCost += getLoss();
            testAccuracy(prediction, labelValidationBatches[i], correctValidation, totalValidation);

        }


//        trainingCost /= (float) sampleTrainBatches.size();
//        validationCost /= (float) sampleValidationBatches.size();
        /*
         * Fill Evaluation Object
         */
        trainingCost/=(float) sampleTrainBatches.size();
        validationCost/= (float) sampleValidationBatches.size();
        correctTraining /= totalTraining;
        correctValidation /= totalValidation;
        evaluation._trainingLoss.push_back(trainingCost );
        evaluation._validationLoss.push_back(validationCost );
        evaluation._trainingAccuracy.push_back(correctTraining );
        evaluation._validationAccuracy.push_back(correctValidation );
//		losses.push_back(std::pair<float, float>(trainingCost, validationCost));
        std::cout << "Epoch: " << k << std::endl;
        std::cout << "Current Training Cost:   " << trainingCost <<" Accuracy:"<<correctTraining<< std::endl;
        std::cout << "Current Validation Cost: " << validationCost <<" Accuracy:"<<correctValidation<< std::endl;
        if (trainingCost < trainingLossThreshold) { break; }

    }


    return evaluation;
}

void NeuralNetwork::testAccuracy(Matrix &results, Matrix &labels, float &correct, float &total) {
    Eigen::Index maxRow, maxCol;
    for (int i = 0; i < results.rows(); i++) {
        results.row(i).maxCoeff(&maxRow, &maxCol);
        int prediction = maxCol;
        labels.row(i).maxCoeff(&maxRow, &maxCol);
        int Actual = maxCol;

        if (prediction == Actual) { correct++; }
        total++;

    }
}


std::vector<Matrix> NeuralNetwork::extractBatchList(std::vector<Matrix> &dataset, int batchSize) {
    if (dataset.size() % batchSize != 0) {
        throw std::runtime_error("The current dataset is not dividable by batchsize:" + batchSize);
    }
    if (dataset.size() == 0) {
        throw std::runtime_error("The dataset is empty");
    }

    std::vector<Matrix> batches;
    batches.reserve(dataset.size() / batchSize);
    int sampleSize = dataset[0].size();
    for (int i = 0; i < dataset.size() / batchSize; i++) {
        Matrix tmp(batchSize, sampleSize);
        for (int s = 0; s < batchSize; s++) {
            tmp.row(s) = dataset[s + i * batchSize];
        }
        batches.emplace_back(tmp);
    }

    return batches;
}