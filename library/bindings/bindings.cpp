//
// Created by pbo on 24.07.19.
//

#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>


#include <InputLayer.hpp>
#include <ConvolutionLayer.hpp>
#include <AbstractLayer.hpp>
#include <MaxPoolLayer.hpp>
#include <DenseLayer.hpp>
#include <LossLayer.hpp>
#include <LogitsLayer.hpp>
#include <NeuralNetwork.hpp>
#include <Graph.hpp>
#include <commonDatatypes.hpp>
#include <LegoDataLoader.hpp>


namespace py = pybind11;


PYBIND11_MODULE(libdl, m) {
	m.doc() = "DeepLearning Lib python module";


/*
 * Definition of Enums
 */
	py::enum_<Optimizer>(m, "Optimizer")
			.value("Adam", Optimizer::Adam);
	py::enum_<ActivationType>(m, "ActivationType")
			.value("ReLu", ActivationType::ReLu)
			.value("Sigmoid", ActivationType::Sigmoid)
			.value("LeakyReLu", ActivationType::LeakyReLu)
			.value("None", ActivationType::None);
	py::enum_<InitializationType>(m, "InitializationType")
			.value("Xavier", InitializationType::Xavier);
	py::enum_<LossType>(m, "LossType")
			.value("CrossEntropy", LossType::CrossEntropy)
			.value("MSE", LossType::MSE);
	/*
	 * Definition Graph
	 */

	py::class_<HyperParameters>(m, "HyperParameters", py::dynamic_attr())
			.def(py::init<int, int, float, Optimizer, float, float>(), py::arg("epochs") = 10, py::arg("batchSize") = 8,
					py::arg("learningRate") = 0.01, py::arg("optimizer") = Optimizer::Adam, py::arg("beta1") = 0.9,
					py::arg("beta2") = 0.999)
			.def("toString", &HyperParameters::toString)
			.def_readwrite("_epochs", &HyperParameters::_epochs);
	py::class_<Graph, std::shared_ptr<Graph>>(m, "Graph").def(py::init<>());
	/*
	 * Definition NeuralNetwor Interface
	 */
	py::class_<NeuralNetwork>(m, "NeuralNetwork")
			.def(py::init<const std::shared_ptr<Graph>, const std::shared_ptr<InputLayer>, const std::shared_ptr<LossLayer>, const HyperParameters>())
			.def("trainBatch", &NeuralNetwork::trainBatch)
			.def("predictBatch", &NeuralNetwork::predictBatch)
			.def("writeVariables", &NeuralNetwork::writeVariables)
			.def("readVariables", &NeuralNetwork::readVariables)
			.def("getLoss", &NeuralNetwork::getLoss)
			.def("train", &NeuralNetwork::train)
			.def("trainAndValidate", &NeuralNetwork::trainAndValidate)
			.def("extractBatchList", &NeuralNetwork::extractBatchList);
	/*
	 * Definition Layers
	 */
	py::class_<AbstractLayer, std::shared_ptr<AbstractLayer>>(m, "AbstractLayer")
			.def(py::init<std::shared_ptr<AbstractLayer>, std::shared_ptr<Graph> >())
			.def(py::init<std::shared_ptr<Graph>>());
	py::class_<InputLayer, AbstractLayer, std::shared_ptr<InputLayer>>(m, "InputLayer")
			.def(py::init<std::shared_ptr<Graph>, int, int, int>());
	py::class_<ConvolutionLayer, AbstractLayer, std::shared_ptr<ConvolutionLayer>>(m, "ConvolutionLayer")
			.def(py::init<std::shared_ptr<AbstractLayer>, std::shared_ptr<Graph>, ActivationType, int, int, int, InitializationType>());
	py::class_<MaxPoolLayer, AbstractLayer, std::shared_ptr<MaxPoolLayer>>(m, "MaxPoolLayer")
			.def(py::init<std::shared_ptr<AbstractLayer>, std::shared_ptr<Graph>, int, int>());
	py::class_<LossLayer, AbstractLayer, std::shared_ptr<LossLayer>>(m, "LossLayer")
			.def(py::init<std::shared_ptr<AbstractLayer>, std::shared_ptr<Graph>, LossType>());
	py::class_<LogitsLayer, AbstractLayer, std::shared_ptr<LogitsLayer>>(m, "LogitsLayer")
			.def(py::init<std::shared_ptr<AbstractLayer>, std::shared_ptr<Graph>, int>());
	py::class_<DenseLayer, AbstractLayer, std::shared_ptr<DenseLayer>>(m, "DenseLayer")
			.def(py::init<std::shared_ptr<AbstractLayer>, std::shared_ptr<Graph>, ActivationType, int, InitializationType>());

	/*
	 * Definition Utils
	 */
	py::class_<LegoDataLoader>(m, "LegoDataLoader")
			.def(py::init<>())
			.def("getData", &LegoDataLoader::getData)
			.def("shuffleData", &LegoDataLoader::shuffleData);

	py::class_<DataSet>(m, "DataSet")
			.def(py::init<std::vector<Matrix>, std::vector<Matrix> >())
			.def(py::init<std::vector<Matrix>, std::vector<Matrix>, std::vector<Matrix>, std::vector<Matrix> >())
			.def(py::init<>())
			.def_readwrite("_trainingSamples", &DataSet::_trainingSamples)
			.def_readwrite("_trainingLabels", &DataSet::_trainingLabels)
			.def_readwrite("_validationSamples", &DataSet::_validationSamples)
			.def_readwrite("_validationLabels", &DataSet::_validationLabels);


	py::class_<TrainingEvaluation>(m, "TrainingEvaluation", py::dynamic_attr())
			.def(py::init<HyperParameters>())
			.def("getTrainingAccuracy", &TrainingEvaluation::getTrainingAccuracy)
			.def("getTrainingLoss", &TrainingEvaluation::getTrainingLoss)
			.def("getValidationAccuracy", &TrainingEvaluation::getValidationAccuracy)
			.def("getValidationLoss", &TrainingEvaluation::getValidationLoss)
			.def("getHyperParameters", &TrainingEvaluation::getHyperParameters)
			.def_readwrite("_trainingAccuracy", &TrainingEvaluation::_trainingAccuracy)
			.def_readwrite("_trainingLoss", &TrainingEvaluation::_trainingLoss)
			.def_readwrite("_validationLoss", &TrainingEvaluation::_validationLoss)
			.def_readwrite("_validationAccuracy", &TrainingEvaluation::_validationAccuracy)
			.def_readwrite("_hyperParameters", &TrainingEvaluation::_hyperParameters);


}