//
// Created by pbo on 18.07.19.
//
#pragma once

#include <Node.hpp>
#include <memory>
#include <DataInitialization.hpp>
#include <Graph.hpp>
#include <OperationsFactory.hpp>
/*!
 * Layer Base class, defines what each Layer needs to implement.
 * The main concept is, that each layer has an output node. this way we can connect all nodes of the computational graph
 * by taking the outputNode of the previous Layer as an input node
 */
class AbstractLayer {
public:

    /*!
     * Creates a new Layer storing the input Layer and the computeGraph
     * using the inputLayer, the input Node, Channels and Batch Size are evaluated
     * @param input
     * @param computeGraph
     */
	AbstractLayer(std::shared_ptr<AbstractLayer> input, std::shared_ptr<Graph> computeGraph);
    /*!
     * Creates a new Layer only storing the computeGraph
     * @param computeGraph
     */
	explicit AbstractLayer(std::shared_ptr<Graph> computeGraph);

	~AbstractLayer() = default;//! default Constructor



private:
	std::shared_ptr<AbstractLayer> _inputLayer; //! the previous Layer
	std::shared_ptr<Node> _inputNode; //! the outputNode of the inputLayer
	std::shared_ptr<Node> _outputNode; //! the outputNode of the current Layer
	int _outputChannels; //!the number of output Channels each sample of the current Layer has
	int _outputSize; //!the outputSize of each sample of the current Layer
	int _inputChannels; //!the number of channels of each sample of the input Layer
	int _batchSize; //! the amount of samples this Layer should compute
	std::shared_ptr<Graph> _computeGraph; //! the computational graph

	/*
	 * Getters & Setters
	 */
public:
    const std::shared_ptr<Node> &getInputNode() const;

    const std::shared_ptr<Node> &getOutputNode() const;

    void setOutputNode(const std::shared_ptr<Node> &outputNode);

    int getOutputChannels() const;

    void setOutputChannels(int outputChannels);

    int getInputChannels() const;

    int getOutputSize() const;

    void setOutputSize(int outputSize);

    int getBatchSize() const;

    void setBatchSize(int batchSize);

    int getInputSize();
    const std::shared_ptr<Graph> &getComputeGraph() const;
};


