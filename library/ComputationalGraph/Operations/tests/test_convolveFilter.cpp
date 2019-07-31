//
// Created by phili on 14.06.2019.
//


#include <MultiplicationOp.hpp>
#include <Placeholder.hpp>
#include <memory>

#include <catch2/catch.hpp>

#include <Parameter.hpp>
#include <ConvolutionOp.hpp>
#include <OperationsFactory.hpp>


TEST_CASE("Im2Col ", "[method]") {

	Eigen::MatrixXf input(1, 32);
	input << 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2,
			1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1;

	Eigen::MatrixXf filter(2, 8);
	filter << 1, 0, 0, 1, 0, 1, 1, 0,
			2, 0, 0, 2,
			0, 2, 2, 0;

	int channel = 2;
	int batchSize = 1;
	int filterSize = 4;
	int origDim = 4;


	SECTION("Stride 1 kernelDim2") {
		Matrix result;
		Eigen::MatrixXf result2;
		Matrix test(8, 9);
		test << 1, 2, 1, 2, 1, 2, 1, 1, 1,
				2, 1, 1, 1, 2, 2, 1, 1, 1,
				2, 1, 2, 1, 1, 1, 2, 2, 2,
				1, 2, 2, 1, 1, 1, 2, 2, 2,
				1, 1, 1, 2, 2, 2, 1, 2, 1,
				1, 1, 1, 2, 2, 2, 2, 1, 2,
				2, 2, 2, 1, 2, 1, 2, 1, 2,
				2, 2, 2, 2, 1, 2, 1, 2, 1;

		ConvolutionOp::im2col(result, input, filterSize, 1, channel, batchSize);
		REQUIRE(result.isApprox(test));
		//We use a col2im which takes into account, that pixels which where used multiple Times are taken into acount
		// twice, Therefore a col2im(im2col(input)) with stride one and dim2 does not return the input. However, if
		// stride is 2 and kernelDim also 2 no pixels are taken twice and col2im(im2col(input)) returns the input as is
		// tested by the Section ("Stride 2, kernelDim 2")
//		ConvolutionOp::col2im(result2,result,filterSize,origDim,1,channel,batchSize);
//		REQUIRE(result2.isApprox(input));
	}
	/*
	 * Stride 2 has to work as well, as no information should be lost here
	 */
	SECTION("Stride 2 kernelDim") {
		Matrix result;
		Matrix result2;

		ConvolutionOp::im2col(result, input, filterSize, 2, channel, batchSize);
		ConvolutionOp::col2im(result2, result, filterSize, origDim, 2, channel, batchSize);

		REQUIRE(result2.isApprox(input));
	}


}

TEST_CASE("Convolution of Parameter ", "[operation]") {

	SECTION("One dimensional filter, one input", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();
		Eigen::MatrixXf img(1, 4);
		img << 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;

		auto X = std::make_shared<Placeholder>(img, 1);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 1);
		graph->computeForward();


		Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
		test << 2, 4, 6, 8;
		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(2, 4);
		img << 1, 2, 3, 4,
				1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 1);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 1);
		graph->computeForward();
		Eigen::MatrixXf test = Eigen::MatrixXf(2, 4);
		test << 2, 4, 6, 8
				, 2, 4, 6, 8;
		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("One dimensional filter, stride 2", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();
		auto graph2 = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 9);
		img << 1, 2, 3, 4, 5, 6, 7, 8, 9;

		Eigen::MatrixXf filter(1, 1);
		filter << 2;
		Eigen::MatrixXf filter2(1, 1);
		filter2 << 2;


		auto X = std::make_shared<Placeholder>(img, 1);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 2);
		graph->computeForward();

		Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
		test << 2, 6, 14, 18;

		REQUIRE(conv->getForward().isApprox(test));

	}
	SECTION("One dimensional filter, stride 3", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 9);
		img << 1, 2, 3, 4, 5, 6, 7, 8, 9;

		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 1);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 3);
		graph->computeForward();
		Eigen::MatrixXf test2 = Eigen::MatrixXf(1, 1);
		test2 << 2;
		REQUIRE(conv->getForward().isApprox(test2));
	}
	SECTION("Multi-dimensional filter, stride 1", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 25);
		img <<
			1, 1, 1, 0, 0,
				0, 1, 1, 1, 0,
				0, 0, 1, 1, 1,
				0, 0, 1, 1, 0,
				0, 1, 1, 0, 0;
		Eigen::MatrixXf filter(1, 9);
		filter <<
			   1, 0, 1,
				0, 1, 0,
				1, 0, 1;

		auto X = std::make_shared<Placeholder>(img, 1);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 1);
		graph->computeForward();

		Eigen::MatrixXf test = Eigen::MatrixXf(1, 9);
		test <<
			 4, 3, 4,
				2, 4, 3,
				2, 3, 4;
		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("MultiChannel", "[Multi_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 2, 1);
		graph->computeForward();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
		test << 5, 10, 15, 20;

		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[Multi_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;


		auto X = std::make_shared<Placeholder>(img, 2);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 2, 1);
		graph->computeForward();
		Eigen::MatrixXf test = Eigen::MatrixXf(2, 4);
		test << 5, 10, 15, 20,
				5, 10, 15, 20;

		REQUIRE(conv->getForward().isApprox(test));
	}
	SECTION("Two Filters and Two Convolutional Layers", "[Multi_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(2, 2);
		filter << 2, 3,
				1, 1;
		Eigen::MatrixXf filter2(1, 2);
		filter2 << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 2, 1);
		std::shared_ptr<ConvolutionOp> conv2 = OperationsFactory::createConvolutionOp(graph, conv, filter2, 2, 1);
		graph->computeForward();

		//Test ForwardPass
		Eigen::MatrixXf testConvF = Eigen::MatrixXf(2, 8);
		testConvF << 5, 10, 15, 20, 2, 4, 6, 8,
				5, 10, 15, 20, 2, 4, 6, 8;
		REQUIRE(conv->getForward().isApprox(testConvF));


		Eigen::MatrixXf testConv2F = Eigen::MatrixXf(2, 4);
		testConv2F << 16, 32, 48, 64,
				16, 32, 48, 64;
		REQUIRE(conv2->getForward().isApprox(testConv2F));

	}

};


TEST_CASE("Backpropagation Parameter ", "[operation]") {

	SECTION("One dimensional filter, one input", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 4);
		img << 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;

		auto X = std::make_shared<Placeholder>(img, 1);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 1);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 1);
		test << 10;

		REQUIRE(conv->getParameter()->getPreviousGradients().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(2, 4);
		img << 1, 2, 3, 4,
				1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 1);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 1);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();

		Eigen::MatrixXf test = Eigen::MatrixXf(1, 1);
		test << 20;
		REQUIRE(conv->getParameter()->getPreviousGradients().isApprox(test));
	}
	SECTION("One dimensional filter, stride =2", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 9);
		img << 1, 2, 3, 4, 5, 6, 7, 8, 9;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 1);

		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 2);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();

		Eigen::MatrixXf test = Eigen::MatrixXf(1, 1);
		test << 1 + 3 + 7 + 9;

		REQUIRE(conv->getParameter()->getPreviousGradients().isApprox(test));


	}
	SECTION("One dimensional filter, stride =3", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 9);
		img << 1, 2, 3, 4, 5, 6, 7, 8, 9;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 1);

		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 3);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();

		Eigen::MatrixXf test2 = Eigen::MatrixXf(1, 1);
		test2 << 1;
		REQUIRE(conv->getParameter()->getPreviousGradients().isApprox(test2));
	}
	SECTION("Multi-dimensional filter, stride 1", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 25);
		img <<
			1, 1, 1, 0, 0,
				0, 1, 1, 1, 0,
				0, 0, 1, 1, 1,
				0, 0, 1, 1, 0,
				0, 1, 1, 0, 0;
		Eigen::MatrixXf filter(1, 9);
		filter <<
			   1, 0, 1,
				0, 1, 0,
				1, 0, 1;

		auto X = std::make_shared<Placeholder>(img, 1);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 1);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();


		Eigen::MatrixXf test = Eigen::MatrixXf(1, 9);
		test <<
			 6, 7, 6,
				4, 7, 7,
				4, 6, 6;
		REQUIRE(conv->getParameter()->getPreviousGradients().isApprox(test));
	}
	SECTION("MultiChannel", "[Multi_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 2, 1);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();


		Eigen::MatrixXf test = Eigen::MatrixXf(1, 2);
		test << 10, 10;

		REQUIRE(conv->getParameter()->getPreviousGradients().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[Multi_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;


		auto X = std::make_shared<Placeholder>(img, 2);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 2, 1);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();


		Eigen::MatrixXf test = Eigen::MatrixXf(1, 2);
		test << 20, 20;
		REQUIRE(conv->getParameter()->getPreviousGradients().isApprox(test));
	}


};


TEST_CASE("Backpropagation Input ", "[operation]") {

	SECTION("One dimensional filter, one input", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 4);
		img << 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;

		auto X = std::make_shared<Placeholder>(img, 1);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 1);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();


		Eigen::MatrixXf test = Eigen::MatrixXf(1, 4);
		test << 2, 2, 2, 2;
		REQUIRE(X->getPreviousGradients().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(2, 4);
		img << 1, 2, 3, 4,
				1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 1);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 1);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();

		Eigen::MatrixXf test = Eigen::MatrixXf(2, 4);
		test << 2, 2, 2, 2,
				2, 2, 2, 2;

		REQUIRE(X->getPreviousGradients().isApprox(test));
	}


	SECTION("One dimensional filter, stride =2", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 9);
		img << 1, 2, 3, 4, 5, 6, 7, 8, 9;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 1);

		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 2);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();

		Eigen::MatrixXf test = Eigen::MatrixXf(1, 9);
		test << 2, 0, 2,
				0, 0, 0,
				2, 0, 2;

		REQUIRE(X->getPreviousGradients().isApprox(test));


	}
	SECTION("One dimensional filter, stride =3", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 9);
		img << 1, 2, 3, 4, 5, 6, 7, 8, 9;
		Eigen::MatrixXf filter(1, 1);
		filter << 2;


		auto X = std::make_shared<Placeholder>(img, 1);

		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 3);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();
		Eigen::MatrixXf test2 = Eigen::MatrixXf(1, 9);
		test2 << 2, 0, 0, 0, 0, 0, 0, 0, 0;
		REQUIRE(X->getPreviousGradients().isApprox(test2));
	}
	SECTION("Multi-dimensional filter, stride 1", "[One_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 25);
		img <<
			1, 1, 1, 0, 0,
				0, 1, 1, 1, 0,
				0, 0, 1, 1, 1,
				0, 0, 1, 1, 0,
				0, 1, 1, 0, 0;
		Eigen::MatrixXf filter(1, 9);
		filter <<
			   1, 0, 1,
				0, 1, 0,
				1, 0, 1;

		auto X = std::make_shared<Placeholder>(img, 1);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 1, 1);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();

		Eigen::MatrixXf test = Eigen::MatrixXf(1, 25);
		test <<
			 1, 1, 2, 1, 1,
				1, 2, 3, 2, 1,
				2, 3, 5, 3, 2,
				1, 2, 3, 2, 1,
				1, 1, 2, 1, 1;
		REQUIRE(X->getPreviousGradients().isApprox(test));
	}
	SECTION("MultiChannel", "[Multi_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(1, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 2, 1);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();
		Eigen::MatrixXf test = Eigen::MatrixXf(1, 8);

		test << 2, 2, 2, 2, 3, 3, 3, 3;

		REQUIRE(X->getPreviousGradients().isApprox(test));
	}
	SECTION("One dimensional filter, miniBatch as input", "[Multi_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(1, 2);
		filter << 2, 3;


		auto X = std::make_shared<Placeholder>(img, 2);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 2, 1);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();


		Eigen::MatrixXf test = Eigen::MatrixXf(2, 8);
		test << 2, 2, 2, 2, 3, 3, 3, 3,
				2, 2, 2, 2, 3, 3, 3, 3;
		REQUIRE(X->getPreviousGradients().isApprox(test));
	}

}

TEST_CASE("Two Convolutional Layers") {


	SECTION("Two Filters and Two Convolutional Layers", "[Multi_Channel_Image]") {
		auto graph = std::make_shared<Graph>();

		Eigen::MatrixXf img(2, 8);
		img << 1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4;
		Eigen::MatrixXf filter(2, 2);
		filter << 2, 3,
				1, 1;
		Eigen::MatrixXf filter2(1, 2);
		filter2 << 2, 3;

		auto X = std::make_shared<Placeholder>(img, 2);
		graph->setInput(X);
		std::shared_ptr<ConvolutionOp> conv = OperationsFactory::createConvolutionOp(graph, X, filter, 2, 1);
		std::shared_ptr<ConvolutionOp> conv2 = OperationsFactory::createConvolutionOp(graph, conv, filter2, 2, 1);
		graph->computeForward();
		graph->computeBackwards();
		graph->updateParameters();

		//Test ForwardPass
		Eigen::MatrixXf testConvF = Eigen::MatrixXf(2, 8);
		testConvF << 5, 10, 15, 20, 2, 4, 6, 8,
				5, 10, 15, 20, 2, 4, 6, 8;
		REQUIRE(conv->getForward().isApprox(testConvF));


		Eigen::MatrixXf testConv2F = Eigen::MatrixXf(2, 4);
		testConv2F << 16, 32, 48, 64,
				16, 32, 48, 64;
		REQUIRE(conv2->getForward().isApprox(testConv2F));

		//Test Backwardpass Weights


		Eigen::MatrixXf testW = Eigen::MatrixXf(2, 2);
		testW << 40, 40,
				60, 60;
		REQUIRE(conv->getParameter()->getPreviousGradients().isApprox(testW));


		Eigen::MatrixXf testW2 = Eigen::MatrixXf(1, 2);
		testW2 << 100, 40;

		REQUIRE(conv2->getParameter()->getPreviousGradients().isApprox(testW2));

		//Test Backwardpass Placeholder

		Eigen::MatrixXf testConvG = Eigen::MatrixXf(2, 8);
		testConvG << 2, 2, 2, 2, 3, 3, 3, 3,
				2, 2, 2, 2, 3, 3, 3, 3;
		Eigen::MatrixXf testCG = Eigen::MatrixXf(2, 8);
		testCG << 7, 7, 7, 7, 9, 9, 9, 9,
				7, 7, 7, 7, 9, 9, 9, 9;
		REQUIRE(conv->getPreviousGradients().isApprox(testConvG));
		REQUIRE(X->getPreviousGradients().isApprox(testCG));

	}

};



