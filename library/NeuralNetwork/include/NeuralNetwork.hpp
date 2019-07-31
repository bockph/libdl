//
// Created by pbo on 24.07.19.
//

#pragma once

#include <InputLayer.hpp>
#include <LossLayer.hpp>
#include <commonDatatypes.hpp>
#include <Graph.hpp>

/*!
 * This class implements the DeepLearning logic to run a training or prediction on a set of Layers/operations, specified by a inputLayer, a lossLayer and a computational graph
 */
class NeuralNetwork {
public:
    /*!
     *  simple construction and initialization of the member variables with provided Parameters
     * @param computeGraph
     * @param inputLayer
     * @param lossLayer
     */
    NeuralNetwork(std::shared_ptr<Graph> computeGraph, std::shared_ptr<InputLayer> inputLayer,
                  std::shared_ptr<LossLayer> lossLayer);
    /*!
     * Trains the Network on the trainingData stored in data
     * @param data
     * @param params
     * @param trainingLossThreshold
     * @return the final Loss
     */
    float train(DataSet &data, HyperParameters params, float trainingLossThreshold = 1);
    /*!
     * Trains the Network on the trainingData stored in data
     * After each epoch the loss and accuracy is evaluated for both trainingSet and Validation Set
     * the results are stored in an TrainingEvaluation object
     *
     * @param data
     * @param params
     * @param trainingLossThreshold this can be used to tell at which cost the training should stop
     * @return Training results
     */
    TrainingEvaluation trainAndValidate(DataSet &data, HyperParameters params, float trainingLossThreshold = 1);
    /*!
     *  Trains the network on a specific batch and the corresponding Labels
     * @param miniBatch one row per sample
     * @param labels one row per sample, one collum for each label. if e.g. sample zero has label one label.row(0)={0,1,0,0,....,0}
     * @param params the HyperParameter object contains everything need to update the Weights, Biases and Filters
     */
    void trainBatch(Matrix &miniBatch, Matrix &labels, HyperParameters &params);

    /*!
     * this executes a simple prediction and returns the predicted labels
     * @param miniBatch
     * @param labels this should be removed in Future, but is still needed as the Loss operation needs an label input
     * @return the Predictions: one row per sample, one collum for each label. if e.g. sample zero has label one label.row(0)={0,1,0,0,....,0}
     */
    Matrix predictBatch(Matrix &miniBatch, Matrix &labels);

    /*!
     * Adds to the variable correct the number of correct predictions
     * Adds to the variable total the number of samples that where predicted
     * accuracy can then be calculated with correct/total
     * @param results
     * @param labels
     * @param correct
     * @param total
     */
    static void testAccuracy(Matrix &results, Matrix &labels, float &correct, float &total);

    /*!
     * Writes the parameter of the current Network
     * @param dir
     * @param networkName
     * @return if succeeded
     */
    bool writeParameters(std::string dir, std::string networkName);

    /*!
    * Initializes the Network with prestored Parameters stored in dir
    * @param dir
    * @param networkName
    * @return if succeeded
    */
    bool readParameters(std::string dir, std::string networkName);

    /*!
     * @return the current Loss
     */
    float getLoss();


    /*!
     *
     * @param dataset a list of samples
     * @param batchSize
     * @return a list of batches
     */
    static std::vector<Matrix> extractBatchList(std::vector<Matrix> &dataset, int batchSize);

private:


    std::shared_ptr<Graph> _computeGraph; //! the computational graph of the network
    std::shared_ptr<InputLayer> _inputLayer; //! the first Layer
    std::shared_ptr<LossLayer> _lossLayer; //! the last layer, implementing the loss calculation logic
    bool _runAchieved; //!sets a flag if a training has been done, if not there wont be a valid loss

};


