//
// Created by pbo on 18.06.19.
//

#include <iostream>
//#include <Eigen/Dense>
#include <Graph.hpp>
#include <Placeholder.hpp>
#include <Operation.hpp>
#include <Session.hpp>
#include <SUM.hpp>
#include <Weight.hpp>
#include <Bias.hpp>
#include <Sigmoid.hpp>
#include <MUL.hpp>
#include <MSE.hpp>
#include <Utils.hpp>
#include <Filter.hpp>
#include <ConvolveFilter.hpp>
#include <ReLu.hpp>
#include <MaxPool.hpp>
#include <Flatten.hpp>
#include <Softmax.hpp>
#include <CrossEntropyLoss.hpp>
#include "mnist/mnist_reader.hpp"



int main() {
    int batch_size = 6;

    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION,batch_size);
    std::cout<<"Dataset:"<<dataset.training_images.size()<<std::endl;


    auto graph = std::make_unique<Graph>();
    //Input Data: with dimensions: Amount of Training Samples x Dimension of training Sample
    Eigen::MatrixXf img(batch_size,28*28);
    /*for ( int i = 0;i< batch_size;i++){
    	Eigen::Matrix<unsigned char,1,784> tmp(dataset.training_images.at(i).data());
    	Eigen::MatrixXf tmp2 = tmp.cast<float>();
    	img.block(i,0,1,784)=tmp2;
    }*/
    img=generateRandomMatrix(0.,1.,6,10*10);

    Eigen::MatrixXf C(batch_size,10);
    C.setZero();
    for ( int i = 0;i< batch_size;i++){
        C(i,dataset.training_labels.at(i))=1;
        /*Eigen::Matrix<unsigned char,1,784> tmp(dataset.training_images.at(i).data());
        Eigen::MatrixXf tmp2 = tmp.cast<float>();
        img.block(i,0,1,784)=tmp2;*/
    }
    C<<1,0,0,0,0,0,0,0,0,0,
            0,1,0,0,0,0,0,0,0,0,
            1,0,0,0,0,0,0,0,0,0,
            0,0,1,0,0,0,0,0,0,0,
            0,0,0,1,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,1;
    auto CN = std::make_shared<Placeholder>(C,0,0);

    Eigen::MatrixXf filter1(8,5*5);
    filter1 = generateRandomMatrix(0.,1.,8,5*5);

    auto outputDim =std::floor((10 - 5) / 1) + 1;
    Eigen::MatrixXf bias1=Eigen::MatrixXf::Zero(1,outputDim*outputDim*8);

    auto X = std::make_shared<Placeholder>(img,10,1);
    auto F = std::make_shared<Filter>(filter1,5,1);
    auto B1 = std::make_shared<Bias>(bias1,8);


    auto conv1 = std::make_shared<ConvolveFilter>(X,F,1);
    auto sum1 = std::make_shared<SUM>(conv1,B1);
    auto relu1  = std::make_shared<ReLu>(sum1);

//convolutional Layer 2
    /*

  auto outputDim2 =std::floor((outputDim - 5) / 1) + 1;
    Eigen::MatrixXf bias2=Eigen::MatrixXf::Zero(1,outputDim2*outputDim2*8);

    auto B2 = std::make_shared<Bias>(bias2,8);

    auto conv2 = std::make_shared<ConvolveFilter>(relu1,F2,1);

    auto sum2 = std::make_shared<SUM>(conv2,B2);
    auto relu2 = std::make_shared<ReLu>(sum2);
*/

    Eigen::MatrixXf filter2(8,3*3*8);
    filter2 = generateRandomMatrix(0.,1.,8,3*3*8);
    auto F2 = std::make_shared<Filter>(filter2,3,8);

//Maxpooling
    auto outputDim3 =std::floor((outputDim - 3) / 1) + 1;
    auto maxPool = std::make_shared<ConvolveFilter>(relu1,2,1);

    auto flattened = std::make_shared<Flatten>(maxPool);
    auto out3DimSQ = std::pow(outputDim3,2)*8;
//Dense Layer 1
    //Weights Hidden Layer 1
    Eigen::MatrixXf mW1 = generateRandomMatrix(0., 1., out3DimSQ, out3DimSQ);
    auto W1 = std::make_shared<Weight>(mW1);
    //Bias
    Eigen::MatrixXf b3 = Eigen::MatrixXf::Zero(1,out3DimSQ);
    auto B3 = std::make_shared<Bias>(b3);

    auto mul1 = std::make_shared<MUL>(flattened, W1);
    auto sum3 = std::make_shared<SUM>(mul1, B3);
    auto relu3 = std::make_shared<ReLu>(sum3);
//Dense Layer 2
    //Weights Hidden Layer 2
    Eigen::MatrixXf mW2 = generateRandomMatrix(0., 1., out3DimSQ,10 );
    auto W2 = std::make_shared<Weight>(mW2);
    //Bias
    Eigen::MatrixXf b4 = Eigen::MatrixXf::Zero(1, 10);
    auto B4 = std::make_shared<Bias>(b4);

    auto mul2 = std::make_shared<MUL>(relu3, W2);
    auto sum4 = std::make_shared<SUM>(mul2, B4);

//    Output/Cost Layer
    auto soft = std::make_shared<Softmax>(sum4,10);
//    auto relu3 = std::make_shared<ReLu>(sum3);



    auto CE = std::make_shared<CrossEntropyLoss>(soft,CN);

    //Create Deep Learning session
    Session session(CE, std::move(graph));

    //session.run() Executes Forward Pass & Backpropagation, Learning Rate is hardcoded at the moment and is 1
    //	std::cout<<"\ncurrent Gradients:\n"<<getCurrentGradients()<<std::endl;
    std::cout<<"\ngo:\n"<<std::endl;


    session.run();
    /*std::cout << "\nFirst Run\n" << std::endl;
    std::cout << "Output:\n" << soft->getForward() << std::endl;*/
    std::cout << "LOSS:\n" << CE->getForward() << std::endl;
    for (int i = 0; i < 10; i++) {
        session.run();
        std::cout<<i<<". run achieved"<<std::endl;
        std::cout << "LOSS:\n" << CE->getForward()(0,0) << std::endl;

    }
    std::cout << "\n Results of Last Run (5002th)\n" << std::endl;
//	std::cout << "Output:\n" << soft->getForward() << std::endl;
    std::cout << "LOSS:\n" << CE->getForward() << std::endl;

    std::cout<<"\nend:\n\n\n"<<std::endl;
/*	session.run();

int test_size =batch_size;
	Eigen::MatrixXf test(test_size,28*28);
	for ( int i = 0;i< test_size;i++){
		Eigen::Matrix<unsigned char,1,784> tmp(dataset.test_images.at(i).data());
		Eigen::MatrixXf tmp2 = tmp.cast<float>();
		img.block(i,0,1,784)=tmp2;
	}
//    img=generateRandomMatrix(0.,1.,6,28*28);
	 auto XTest = std::make_shared<Placeholder>(test,28,1);
	conv1->setInputA(XTest);

	Eigen::MatrixXf CTest(test_size,10);
	C.setZero();
	for ( int i = 0;i< test_size;i++){
		C(i,dataset.test_labels.at(i))=1;
		*//*Eigen::Matrix<unsigned char,1,784> tmp(dataset.training_images.at(i).data());
		Eigen::MatrixXf tmp2 = tmp.cast<float>();
		img.block(i,0,1,784)=tmp2;*//*
	}
	session.run();
	Eigen::MatrixXf::Index maxRow, maxCol;
	float correct=0;
	float wrong=0;
	for(int i =0;i<test_size;i++){
		soft->getForward().block(i,0,1,10).maxCoeff(&maxRow,&maxCol);
		int p = maxCol;
		C.block(i,0,1,10).maxCoeff(&maxRow,&maxCol);
		int A = maxCol;

		if(p==A)correct++;
		else wrong++;
//		std::cout<<"P: "<<p<<std::endl;
//		std::cout<<"A: "<<A<<std::endl;

	}

	std::cout<<"Amount Correct:"<<correct<<"Percentage :"<<correct/(float)test_size<<std::endl;
	std::cout<<"Amount Wrong:"<<wrong<<"Percentage :"<<wrong/(float)test_size<<std::endl;*/





//
    //    std::cout << "First Run" << std::endl;
//    std::cout << "Output:\n" << sig2->getForward() << std::endl;
//    std::cout << "LOSS:\n" << mse->getForward() << std::endl;
//    for (int i = 0; i < 5000; i++) {
//        session.run();
//    }
//    session.run();
//    std::cout << " Results of Last Run (5002th)" << std::endl;
//    std::cout << "Output:\n" << sig2->getForward() << std::endl;
//    std::cout << "LOSS:\n" << mse->getForward() << std::endl;
}