//
// Created by pbo on 18.06.19.
//

#include <iostream>
#include <Placeholder.hpp>
#include <Operation.hpp>
#include <Session.hpp>
#include <SummationOp.hpp>
#include <MultiplicationOp.hpp>
#include <DataInitialization.hpp>
#include <Variable.hpp>
#include <ConvolveFilterIM2COL.hpp>
#include <ReLuOp.hpp>
#include <MaxPoolOp.hpp>
#include <SoftmaxOp.hpp>
#include <CrossEntropyOp.hpp>
#include "mnist/mnist_reader.hpp"
#include <mnist/mnist_utils.hpp>
#
#include <IO.hpp>
#include <SigmoidOP.hpp>
#include <InputLayer.hpp>
#include <ConvolutionLayer.hpp>
#include <AbstractLayer.hpp>
#include <MaxPoolLayer.hpp>
#include <DenseLayer.hpp>
#include <LossLayer.hpp>
#include <LogitsLayer.hpp>

void getBatches(int batch_size, int amountBatches, std::vector<Eigen::MatrixXf>& training_data, std::vector<Eigen::MatrixXf>& label_data,bool trainData =true){

    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION,
            		batch_size*amountBatches);
    mnist::normalize_dataset(dataset);

    training_data.clear();
    label_data.clear();

    for(int j = 0;j<amountBatches;j++){
        Eigen::MatrixXf img(batch_size,28*28);
        for ( int i = 0;i< batch_size;i++){
            if(trainData){
                Eigen::Matrix<unsigned char,1,784> tmp(dataset.training_images.at(i+j*batch_size).data());
                Eigen::MatrixXf tmp2 = tmp.cast<float>();
                img.block(i,0,1,784)=tmp2;
            }else{
            	if(i+j*batch_size<5000){
					Eigen::Matrix<unsigned char,1,784> tmp(dataset.test_images.at(i+j*batch_size).data());
					Eigen::MatrixXf tmp2 = tmp.cast<float>();
					img.block(i,0,1,784)=tmp2;
            	}

            }

        }


        training_data.push_back(img);
        Eigen::MatrixXf C(batch_size,10);
        C.setZero();
        for ( int i = 0;i< batch_size;i++){
            if(trainData)
                C(i,dataset.training_labels.at(i+j*batch_size))=1;
            else{
				if(i+j*batch_size<5000){
					C(i,dataset.test_labels.at(i+j*batch_size))=1;
				}
            }


        }

        label_data.push_back(C);
    }




}
float train(std::vector<Eigen::MatrixXf>& params,float &correct,float &total, bool train =true){
    /*
     * params = [img,label,f1,f2,w3,w4,b1,b2,b3,b4]
     */


    //Create InputLayer
    auto input = std::make_shared<InputLayer>(params[0],28,1);
    std::shared_ptr<Placeholder> CN = std::make_shared<Placeholder>(params[1],0,0);

//Convolutional Layer 1

    auto convolution1=std::make_shared<ConvolutionLayer>(input,AbstractLayer::ActivationFunction::Sigmoid,8,5,1,AbstractLayer::InitializationType::Xavier);



//convolutional Layer 2
    auto convolution2=std::make_shared<ConvolutionLayer>(convolution1,AbstractLayer::ActivationFunction::ReLu,8,5,1,AbstractLayer::InitializationType::Xavier);
//Maxpooling
    auto maxPool =std::make_shared<MaxPoolLayer>(convolution2,2,2);

//Dense Layer 1
    auto dense1 = std::make_shared<DenseLayer>(maxPool,AbstractLayer::ActivationFunction::ReLu,128,AbstractLayer::InitializationType::Xavier);

//Dense Layer 2
    auto dense2 = std::make_shared<DenseLayer>(dense1,AbstractLayer::ActivationFunction::None,10,AbstractLayer::InitializationType::Xavier);


//Logits Layer
    auto logits = std::make_shared<LogitsLayer>(dense2,10);

//    Cost Layer
    auto loss = std::make_shared<LossLayer>(logits,params[1],AbstractLayer::LossType::CrossEntropy);

    //Create Deep Learning session
    Session session(loss->getOutputNode());





    if(train){

		session.run();
        std::cout<<loss->getLoss()<<std::endl;
//        std::cout << "Total forward" << session.getForwardTime()<<std::endl;
//        std::cout << "Total backwards" << session.getBackwardsTime()<<std::endl;
//
//        std::cout << "Convolution 1: Total forward: " << conv1->getForwardTime()<<"Percentage: "<<(float)conv1->getForwardTime()/(float)session.getForwardTime()<<std::endl;
//        std::cout << "Convolution 1: Total backwards: " << conv1->getBackwardsTime()<<"Percentage: "<<(float)conv1->getBackwardsTime()/(float)session.getBackwardsTime()<<std::endl;
//
//        std::cout << "Convolution 2 Total forward: " << conv2->getForwardTime()<<"Percentage: "<<(float)conv2->getForwardTime()/(float)session.getForwardTime()<<std::endl;
//        std::cout << "Convolution 2: Total backwards: " << conv2->getBackwardsTime()<<"Percentage: "<<(float)conv2->getBackwardsTime()/(float)session.getBackwardsTime()<<std::endl;
//
//        std::cout << "Maxpool: Total forward: " << maxPool->getForwardTime()<<"Percentage: "<<(float)maxPool->getForwardTime()/(float)session.getForwardTime()<<std::endl;
//        std::cout << "MaxPoolOp: Total backwards: " << maxPool->getBackwardsTime()<<"Percentage: "<<(float)maxPool->getBackwardsTime()/(float)session.getBackwardsTime()<<std::endl;
//        params[2] = F1->getForward();
//        params[3] = F2->getForward();
//        params[4] = W1->getForward();
//        params[5] = W2->getForward();
//        params[6] = B1->getForward();
//        params[7] = B2->getForward();
//        params[8] = B3->getForward();
//        params[9] = B4->getForward();
    }else{
        session.run();

        Eigen::MatrixXf::Index maxRow, maxCol;

        for(int i =0;i<params[0].rows();i++){
            logits->getOutputNode()->getForward().block(i,0,1,10).maxCoeff(&maxRow,&maxCol);
            int p = maxCol;
            params[1].block(i,0,1,10).maxCoeff(&maxRow,&maxCol);
            int A = maxCol;

            if(p==A)correct++;
            total++;

        }

    }


    return loss->getLoss();


}



int main() {
    Eigen::initParallel();
    /*
     * batch_size: if this is changed '#define BATCH_SIZE' in Node.hpp has to be changed as well
     * epochs: sets the amount of epochs for training, to big values in combination with a big 'amount_batches' can lead to OutOfMemory Error
     * amount_batches: 'batch_size*amount_batches' gives the total amount of samples
     * trainModel: defines if the model should be trained with the aboved set parameters
     * testModel: defines if the model should be tested (using the MNIST test_data)
     * writeWeights: if set the trained Weights  are written to Source_Directory/WeightDeposit
     * readWeights: if set (and Weights have already been Written once) weights are initialized with weights from Source_Directory/WeightDeposit
     */
	int batch_size = 32;
	int epochs =10;
	int amount_batches = 10;
	bool trainModel = true;
	bool testModel =true;
    bool writeWeights = false;
    bool readWeights =false;





	float correct,total;
    std::vector<Eigen::MatrixXf> training_data,training_label;
    std::vector<Eigen::MatrixXf> test_data,test_label;

    getBatches(batch_size,amount_batches,training_data,training_label);
    getBatches(batch_size,amount_batches,test_data,test_label,false);




    //Initialize Weights & Bias & Variable


    std::vector<Eigen::MatrixXf> params ={training_data[0],training_label[0]};
    if(readWeights){
        read_binary(WEIGHT_DEPOSIT"/f1.txt",params[2]);
        read_binary(WEIGHT_DEPOSIT"/f2.txt",params[3]);
        read_binary(WEIGHT_DEPOSIT"/w1.txt",params[4]);
        read_binary(WEIGHT_DEPOSIT"/w2.txt",params[5]);
        read_binary(WEIGHT_DEPOSIT"/b1.txt",params[6]);
        read_binary(WEIGHT_DEPOSIT"/b2.txt",params[7]);
        read_binary(WEIGHT_DEPOSIT"/b3.txt",params[8]);
        read_binary(WEIGHT_DEPOSIT"/b4.txt",params[9]);
    }



    //Create InputLayer
    auto inputLayer = std::make_shared<InputLayer>(params[0],28,1);
    std::shared_ptr<Placeholder> CN = std::make_shared<Placeholder>(params[1],0,0);

//Convolutional Layer 1

    auto convolution1=std::make_shared<ConvolutionLayer>(inputLayer,AbstractLayer::ActivationFunction::ReLu,32,5,1,AbstractLayer::InitializationType::Xavier);

    auto maxPool2 =std::make_shared<MaxPoolLayer>(convolution1,2,2);


//convolutional Layer 2
    auto convolution2=std::make_shared<ConvolutionLayer>(maxPool2,AbstractLayer::ActivationFunction::ReLu,64,5,1,AbstractLayer::InitializationType::Xavier);
//Maxpooling
    auto maxPool =std::make_shared<MaxPoolLayer>(convolution2,2,2);

//Dense Layer 1
    auto dense1 = std::make_shared<DenseLayer>(maxPool,AbstractLayer::ActivationFunction::ReLu,1024,AbstractLayer::InitializationType::Xavier);

//Dense Layer 2
    auto dense2 = std::make_shared<DenseLayer>(dense1,AbstractLayer::ActivationFunction::None,10,AbstractLayer::InitializationType::Xavier);


//Logits Layer
    auto logits = std::make_shared<LogitsLayer>(dense2,10);

//    Cost Layer
    auto loss = std::make_shared<LossLayer>(logits,params[1],AbstractLayer::LossType::CrossEntropy);

    //Create Deep Learning session
    Session session(loss->getOutputNode());


    /*
     * Train Data
     */
    if(trainModel){
        float cost =0;
        for(int k = 0;k<epochs;k++){
            cost=0;
            for(int i = 0;i<amount_batches;i++){
                params[0] = training_data[i];
                params[1] = training_label[i];
                inputLayer->updateX(training_data[i]);
                loss->updateLabels(training_label[i]);
                session.run();

                cost+=loss->getLoss();

            }
            cost/=(float)amount_batches;
            std::cout<<"Current Cost:"<<cost<<" Round: "<<k<<std::endl;
            if(cost<1)break;

        }
    }

    if(writeWeights){
        write_binary(WEIGHT_DEPOSIT"/f1.txt",params[2]);
        write_binary(WEIGHT_DEPOSIT"/f2.txt",params[3]);
        write_binary(WEIGHT_DEPOSIT"/w1.txt",params[4]);
        write_binary(WEIGHT_DEPOSIT"/w2.txt",params[5]);
        write_binary(WEIGHT_DEPOSIT"/b1.txt",params[6]);
        write_binary(WEIGHT_DEPOSIT"/b2.txt",params[7]);
        write_binary(WEIGHT_DEPOSIT"/b3.txt",params[8]);
        write_binary(WEIGHT_DEPOSIT"/b4.txt",params[9]);
    }

    if(testModel){
        for(int b = 0;b<amount_batches;b++){
//            params[0] = test_data[i];
//            params[1] = test_label[i];
//            train(params,correct,total,false);
            inputLayer->updateX(test_data[b]);
            loss->updateLabels(test_label[b]);
            session.run();

            Eigen::MatrixXf::Index maxRow, maxCol;

            for(int i =0;i<test_data[b].rows();i++){
                logits->getOutputNode()->getForward().block(i,0,1,10).maxCoeff(&maxRow,&maxCol);
                int p = maxCol;
                test_label[b].block(i,0,1,10).maxCoeff(&maxRow,&maxCol);
                int A = maxCol;

                if(p==A)correct++;
                total++;

            }


        }
        std::cout<<"Amount Correct: "<<correct<<"\nAmount Wrong: "<<total-correct<<"\nPercentage Correct: "<<correct/(float)
                total<<std::endl;
    }




}