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


int main() {
    auto graph = std::make_unique<Graph>();
    //Input Data: with dimensions: Amount of Training Samples x Dimension of training Sample
    Eigen::MatrixXf img(1,28*28);
    img=generateRandomMatrix(0.,1.,1,28*28);

    Eigen::MatrixXf filter1(8,5*5);
    filter1 = generateRandomMatrix(0.,1.,8,5*5);

    auto outputDim =std::floor((28 - 5) / 1) + 1;
    Eigen::MatrixXf bias1=Eigen::MatrixXf::Zero(1,outputDim*outputDim*8);

    auto X = std::make_shared<Placeholder>(img,28,1);
    auto F = std::make_shared<Filter>(filter1,5,1);
    auto B1 = std::make_shared<Bias>(bias1);


    auto conv1 = std::make_shared<ConvolveFilter>(X,F,1);
    auto sum1 = std::make_shared<SUM>(conv1,B1);
    auto relu1  = std::make_shared<ReLu>(sum1);

//convolutional Layer 2
    Eigen::MatrixXf filter2(8,5*5*8);
    filter2 = generateRandomMatrix(0.,1.,1,5*5*8);

    auto outputDim2 =std::floor((outputDim - 5) / 1) + 1;
    Eigen::MatrixXf bias2=Eigen::MatrixXf::Zero(1,outputDim2*outputDim2);

    auto F2 = std::make_shared<Filter>(filter2,5,8);
    auto B2 = std::make_shared<Bias>(bias2);


    auto conv2 = std::make_shared<ConvolveFilter>(relu1,F2,1);
    auto sum2 = std::make_shared<SUM>(conv2,B2);
    auto relu2 = std::make_shared<ReLu>(sum2);

//Maxpooling
    auto outputDim3 =std::floor((outputDim2 - 2) / 2) + 1;
    auto maxPool = std::make_shared<MaxPool>(relu2,2,2);


//Dense Layer 1
    //Weights Hidden Layer 1
    Eigen::MatrixXf mW1 = generateRandomMatrix(0., 1., outputDim3*outputDim3, outputDim3*outputDim3);
    auto W1 = std::make_shared<Weight>(mW1);
    //Bias
    Eigen::MatrixXf b3 = Eigen::MatrixXf::Zero(1, outputDim3*outputDim3);
    auto B3 = std::make_shared<Bias>(b3);

    auto mul1 = std::make_shared<MUL>(maxPool, W1);
    auto sum3 = std::make_shared<SUM>(mul1, B3);
    auto relu3 = std::make_shared<ReLu>(sum3);
//Dense Layer 2
    //Weights Hidden Layer 2
    Eigen::MatrixXf mW2 = generateRandomMatrix(0., 1., outputDim3*outputDim3,10 );
    auto W2 = std::make_shared<Weight>(mW2);
    //Bias
    Eigen::MatrixXf b4 = Eigen::MatrixXf::Zero(1, 10);
    auto B4 = std::make_shared<Bias>(b4);

    auto mul2 = std::make_shared<MUL>(relu3, W2);
    auto sum4 = std::make_shared<SUM>(mul2, B4);
//    auto relu3 = std::make_shared<ReLu>(sum3);


    //Create Deep Learning session
    Session session(sum4, std::move(graph));

    //session.run() Executes Forward Pass & Backpropagation, Learning Rate is hardcoded at the moment and is 1
    session.run();
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