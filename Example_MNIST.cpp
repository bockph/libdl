//
// Created by pbo on 18.06.19.
//

#include <iostream>
#include <Graph.hpp>
#include <Placeholder.hpp>
#include <Operation.hpp>
#include <Session.hpp>
#include <SUM.hpp>
#include <Weight.hpp>
#include <Bias.hpp>
#include <MUL.hpp>
#include <Utils.hpp>
#include <Filter.hpp>
#include <ConvolveFilter.hpp>
#include <ReLu.hpp>
#include <MaxPool.hpp>
#include <Flatten.hpp>
#include <Softmax.hpp>
#include <CrossEntropyLoss.hpp>
#include "mnist/mnist_reader.hpp"
#include <mnist/mnist_utils.hpp>
#include <Sigmoid.hpp>

namespace Eigen{
    void write_binary(const char* filename, const MatrixXf& matrix){
        std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        typename MatrixXf::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename MatrixXf::Index));
        out.write((char*) (&cols), sizeof(typename MatrixXf::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename MatrixXf::Scalar) );
        out.close();
    }
    void read_binary(const char* filename, MatrixXf& matrix){
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        typename MatrixXf::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename MatrixXf::Index));
        in.read((char*) (&cols),sizeof(typename MatrixXf::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename MatrixXf::Scalar) );
        in.close();
    }
} // Eigen::

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

    auto graph = std::make_unique<Graph>();
    auto X = std::make_shared<Placeholder>(params[0],28,1);
    auto CN = std::make_shared<Placeholder>(params[1],0,0);

//Convolutional Layer 1
    auto F1 = std::make_shared<Filter>(params[2],5,1);
    auto B1 = std::make_shared<Bias>(params[6],8);

    auto conv1 = std::make_shared<ConvolveFilter>(X,F1,1);
    auto sum1 = std::make_shared<SUM>(conv1,B1);
    auto relu1  = std::make_shared<Sigmoid>(sum1);

//convolutional Layer 2
    auto F2 = std::make_shared<Filter>(params[3],5,8);
    auto B2 = std::make_shared<Bias>(params[7],8);

    auto conv2 = std::make_shared<ConvolveFilter>(relu1,F2,1);
    auto sum2 = std::make_shared<SUM>(conv2,B2);
    auto relu2 = std::make_shared<ReLu>(sum2);

//Maxpooling
    auto maxPool = std::make_shared<MaxPool>(relu2,2,2);
    auto flattened = std::make_shared<Flatten>(maxPool);

//Dense Layer 1
    auto W1 = std::make_shared<Weight>(params[4]);
    auto B3 = std::make_shared<Bias>(params[8]);

    auto mul1 = std::make_shared<MUL>(flattened, W1);
	auto sum3 = std::make_shared<SUM>(mul1, B3);

	auto relu3 = std::make_shared<ReLu>(sum3);

//Dense Layer 2
    auto W2 = std::make_shared<Weight>(params[5]);
    auto B4 = std::make_shared<Bias>(params[9]);

    auto mul2 = std::make_shared<MUL>(relu3, W2);
    auto sum4 = std::make_shared<SUM>(mul2, B4);

//    Output/Cost Layer
    auto soft = std::make_shared<Softmax>(sum4,10);
    auto CE = std::make_shared<CrossEntropyLoss>(soft,CN);

    //Create Deep Learning session
    Session session(CE, std::move(graph));


    if(train){

		session.run();

        params[2] = F1->getForward();
        params[3] = F2->getForward();
        params[4] = W1->getForward();
        params[5] = W2->getForward();
        params[6] = B1->getForward();
        params[7] = B2->getForward();
        params[8] = B3->getForward();
        params[9] = B4->getForward();
    }else{
        session.run();

        Eigen::MatrixXf::Index maxRow, maxCol;

        for(int i =0;i<params[0].rows();i++){
            soft->getForward().block(i,0,1,10).maxCoeff(&maxRow,&maxCol);
            int p = maxCol;
            params[1].block(i,0,1,10).maxCoeff(&maxRow,&maxCol);
            int A = maxCol;

            if(p==A)correct++;
            total++;

        }

    }


    return CE->getForward()(0,0);


}



int main() {
    /*
     * batch_size: Don't change this unless one changes as well #define BATCH_SIZE in Node.hpp
     * epochs: self explanatory, but to big values in combination with amount_batches can lead to OutOfMemory
     * amount_batches: batch_size*amount_batches gives the total amount of samples
     * trainModel: if training should be done with TRAINING data
     * testModel: if prediction should be done with TEST data
     * writeWeights: writes the trained Weights to Source_Directory/WeightDeposit
     * readWeights: if Weights have already been Written once (check WeightDeposit Folder), this can be set to true --> no random initialization
     */
	int batch_size = 8;
	int epochs =10;
	int amount_batches =100;
	bool trainModel = false;
	bool testModel =true;
    bool writeWeights = false;
    bool readWeights =true;





	float correct,total;
    std::vector<Eigen::MatrixXf> training_data,training_label;
    std::vector<Eigen::MatrixXf> test_data,test_label;

    getBatches(batch_size,amount_batches,training_data,training_label);
    getBatches(batch_size,amount_batches,test_data,test_label,false);

    //precalculate Dimensions needed for Weights
    auto outputDim =std::floor((28 - 5) / 1) + 1;
    auto outputDim2 =std::floor((outputDim - 5) / 1) + 1;
    auto outputDim3 =std::floor((outputDim2 - 2) / 2) + 1;
    auto out3DimSQ = std::pow(outputDim3,2)*8;


    //Initialize Weights & Bias & Filter

	Eigen::MatrixXf filter1 = generateRandomMatrix(0,.1,8,5*5);
	Eigen::MatrixXf filter2 = generateRandomMatrix(0.,.1,8,5*5*8);

	Eigen::MatrixXf W1 = generateRandomMatrix(0., .1, out3DimSQ, 128);
	Eigen::MatrixXf W2 = generateRandomMatrix(0., .1,128, 10 );

    Eigen::MatrixXf b1=Eigen::MatrixXf::Zero(batch_size,outputDim*outputDim*8);
    Eigen::MatrixXf b2=Eigen::MatrixXf::Zero(batch_size,outputDim2*outputDim2*8);
    Eigen::MatrixXf b3 = Eigen::MatrixXf::Zero(batch_size,128);
    Eigen::MatrixXf b4 = Eigen::MatrixXf::Zero(batch_size, 10);
    std::vector<Eigen::MatrixXf> params ={training_data[0],training_label[0],filter1,filter2,W1,W2,b1,b2,b3,b4};
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

                cost+=train(params,correct,total);

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
        for(int i = 0;i<amount_batches;i++){
            params[0] = test_data[i];
            params[1] = test_label[i];
            train(params,correct,total,false);

        }
        std::cout<<"Amount Correct: "<<correct<<"Amount Wrong: "<<total-correct<<"Percentage: "<<correct/(float)
                total<<std::endl;
    }




}