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
#include <mnist/mnist_utils.hpp>

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


	std::vector<std::vector<float>> data={{1,0},{0,1},{1,1},{0,0}};
	std::vector<int> label={0,0,1,1};

	training_data.clear();
	label_data.clear();

	for(int j = 0;j<amountBatches;j++){
		Eigen::MatrixXf img(batch_size,2);
		for ( int i = 0;i< batch_size;i++){
				Eigen::Matrix<float,1,2> tmp(data.at(i+j*batch_size).data());
				img.block(i,0,1,2)=tmp;


		}


		training_data.push_back(img);
		Eigen::MatrixXf C(batch_size,2);
		C.setZero();
		for ( int i = 0;i< batch_size;i++){
				C(i,label.at(i+j*batch_size))=1;
		}

		label_data.push_back(C);
	}




}
void train(std::vector<Eigen::MatrixXf>& params,float &correct,float &total, bool train =true){
	/*
	 * params = [img,label,f1,f2,w3,w4,b1,b2,b3,b4]
	 */

	auto graph = std::make_unique<Graph>();
	auto X = std::make_shared<Placeholder>(params[0],0,0);
	auto CN = std::make_shared<Placeholder>(params[1],0,0);

	auto W = std::make_shared<Weight>(params[2]);
	auto W2 = std::make_shared<Weight>(params[3]);
	auto B1 = std::make_shared<Bias>(params[4],0);
	auto B2 = std::make_shared<Bias>(params[5],0);

	//create First hidden Layer
	auto mul = std::make_shared<MUL>(X, W);
	auto sum = std::make_shared<SUM>(mul, B1);
	auto sig1 = std::make_shared<Sigmoid>(sum);

	//create output layer
	auto mul2 = std::make_shared<MUL>(sig1, W2);
	auto sum2 = std::make_shared<SUM>(mul2, B2);
//	auto sig2 = std::make_shared<Sigmoid>(sum2);
	auto soft = std::make_shared<Softmax>(sum2,2);

	//create Loss function
	auto mse = std::make_shared<CrossEntropyLoss>(soft, CN);

	//Create Deep Learning session
	Session session(mse, std::move(graph));


	if(train){
		/* for (int i = 0; i < 500; i++) {
 //            std::cout<<i<<". run achieved"<<std::endl;
 //            std::cout << "LOSS:\n" << CE->getForward()(0,0) << std::endl;
			 if(CE->getForward()(0,0)<0.5)break;

		 }*/
		session.run();

		std::cout << "LOSS:\n" << mse->getForward()(0,0) << std::endl;

//        std::cout << "\n Results of Last Run \n" << std::endl;
        std::cout << "Output:\n" << soft->getForward() << std::endl;
//        std::cout << "LOSS:\n" << CE->getForward()(0,0) << std::endl;
//        std::cout<<"\nExpected:\n\n\n"<<CN->getForward()<<std::endl;
		params[2] = W->getForward();
		params[3] = W2->getForward();
		params[4] = B1->getForward();
		params[5] = B2->getForward();
//		params[6] = B1->getForward();
//		params[7] = B2->getForward();
//		params[8] = B3->getForward();
//		params[9] = B4->getForward();
		/*   write_binary("/home/pbo/Schreibtisch/StoredValues/f1.txt",params[2]);
		   write_binary("/home/pbo/Schreibtisch/StoredValues/f2.txt",params[3]);
		   write_binary("/home/pbo/Schreibtisch/StoredValues/w1.txt",params[4]);
		   write_binary("/home/pbo/Schreibtisch/StoredValues/w2.txt",params[5]);
		   write_binary("/home/pbo/Schreibtisch/StoredValues/b1.txt",params[6]);
		   write_binary("/home/pbo/Schreibtisch/StoredValues/b2.txt",params[7]);
		   write_binary("/home/pbo/Schreibtisch/StoredValues/b3.txt",params[8]);
		   write_binary("/home/pbo/Schreibtisch/StoredValues/b4.txt",params[9]);*/
	}else{
		session.run();
		Eigen::MatrixXf::Index maxRow, maxCol;
		/*float correct=0;
		float wrong=0;*/
		for(int i =0;i<params[0].rows();i++){
			soft->getForward().block(i,0,1,2).maxCoeff(&maxRow,&maxCol);
			int p = maxCol;
			params[1].block(i,0,1,2).maxCoeff(&maxRow,&maxCol);
			int A = maxCol;

			if(p==A)correct++;
			total++;
//            else wrong++;


		}

//        std::cout<<"Amount Correct: "<<correct<<"Amount Wrong: "<<wrong<<"Percentage: "<<correct/(float)params[0].rows()<<std::endl;
//        std::cout<<"Amount Wrong:"<<wrong<<"Percentage :"<<wrong/(float)params[0].rows()<<std::endl;
	}




}



int main() {
	int batch_size = 1;
	int epochs =100000;
	int amount_batches =4;
	float correct,total;
	std::vector<Eigen::MatrixXf> training_data,training_label;
	std::vector<Eigen::MatrixXf> test_data,test_label;

	getBatches(batch_size,amount_batches,training_data,training_label);
	getBatches(batch_size,amount_batches,test_data,test_label,false);

	//Initialize Weights & Bias & Filter
	auto outputDim =std::floor((28 - 5) / 1) + 1;
	auto outputDim2 =std::floor((outputDim - 5) / 1) + 1;
	auto outputDim3 =std::floor((outputDim2 - 2) / 2) + 1;
	auto out3DimSQ = std::pow(outputDim3,2)*8;


	/*Eigen::MatrixXf filter1 = initializeFilter(8,5*5);// generateRandomMatrix(0,.1,8,5*5);
	Eigen::MatrixXf filter2 = initializeFilter(8,5*5*8);//generateRandomMatrix(0.,.1,8,5*5*8);*/



	Eigen::MatrixXf W1 = generateRandomMatrix(0., .1, 2, 2);
	Eigen::MatrixXf W2 = generateRandomMatrix(0., .1,2, 2 );


	Eigen::MatrixXf b1 = Eigen::MatrixXf::Zero(1,2);
	Eigen::MatrixXf b2 = Eigen::MatrixXf::Zero(1, 2);
	std::vector<Eigen::MatrixXf> params ={training_data[0],training_label[0],W1,W2,b1,b2};
	/* read_binary("/home/pbo/Schreibtisch/StoredValues/f1.txt",params[2]);
	 read_binary("/home/pbo/Schreibtisch/StoredValues/f2.txt",params[3]);
	 read_binary("/home/pbo/Schreibtisch/StoredValues/w1.txt",params[4]);
	 read_binary("/home/pbo/Schreibtisch/StoredValues/w2.txt",params[5]);
	 read_binary("/home/pbo/Schreibtisch/StoredValues/b1.txt",params[6]);
	 read_binary("/home/pbo/Schreibtisch/StoredValues/b2.txt",params[7]);
	 read_binary("/home/pbo/Schreibtisch/StoredValues/b3.txt",params[8]);
	 read_binary("/home/pbo/Schreibtisch/StoredValues/b4.txt",params[9]);*/
	for(int k = 0;k<epochs;k++){
		for(int i = 0;i<amount_batches;i++){
			params[0] = training_data[i];
			params[1] = training_label[i];
			/*if(i!=0){

			}*/


			train(params,correct,total);

		}
		std::cout<<"Left to go: "<<epochs-k<<std::endl;

	}

	write_binary("/home/pbo/Schreibtisch/StoredValues/f1.txt",params[2]);
	write_binary("/home/pbo/Schreibtisch/StoredValues/f2.txt",params[3]);
	write_binary("/home/pbo/Schreibtisch/StoredValues/w1.txt",params[4]);
	write_binary("/home/pbo/Schreibtisch/StoredValues/w2.txt",params[5]);
	write_binary("/home/pbo/Schreibtisch/StoredValues/b1.txt",params[6]);
	write_binary("/home/pbo/Schreibtisch/StoredValues/b2.txt",params[7]);
	write_binary("/home/pbo/Schreibtisch/StoredValues/b3.txt",params[8]);
	write_binary("/home/pbo/Schreibtisch/StoredValues/b4.txt",params[9]);
//    write_binary("/home/pbo/Schreibtisch/StoredValues/b4.txt",params[9]);
//    std::cout<<"Actual:\n"<<params[9]<<std::endl;
//    params[9].setZero();
//    std::cout<<"Actual:\n"<<params[9]<<std::endl;
//
//    read_binary("/home/pbo/Schreibtisch/StoredValues/b4.txt",params[9]);
//    std::cout<<"Read:\n"<<params[9]<<std::endl;
	for(int i = 0;i<amount_batches;i++){
		/*read_binary("/home/pbo/Schreibtisch/StoredValues/f1.txt",params[2]);
		read_binary("/home/pbo/Schreibtisch/StoredValues/f2.txt",params[3]);
		read_binary("/home/pbo/Schreibtisch/StoredValues/w1.txt",params[4]);
		read_binary("/home/pbo/Schreibtisch/StoredValues/w2.txt",params[5]);
		read_binary("/home/pbo/Schreibtisch/StoredValues/b1.txt",params[6]);
		read_binary("/home/pbo/Schreibtisch/StoredValues/b2.txt",params[7]);
		read_binary("/home/pbo/Schreibtisch/StoredValues/b3.txt",params[8]);
		read_binary("/home/pbo/Schreibtisch/StoredValues/b4.txt",params[9]);*/
		params[0] = training_data[i];
		params[1] = training_label[i];
		train(params,correct,total,false);

	}
	std::cout<<"Amount Correct: "<<correct<<"Amount Wrong: "<<total<<"Percentage: "<<correct/(float)
			total<<std::endl;



}