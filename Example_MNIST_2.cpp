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

	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
			mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION,
					batch_size*amountBatches+1);
//	mnist::normalize_dataset(dataset);

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
void train(std::vector<Eigen::MatrixXf>& params,float &correct,float &total, bool train =true){
	/*
	 * params = [img,label,f1,f2,w3,w4,b1,b2,b3,b4]
	 */

	auto graph = std::make_unique<Graph>();
	auto X = std::make_shared<Placeholder>(params[0],28,1);
	auto CN = std::make_shared<Placeholder>(params[1],0,0);


	auto W = std::make_shared<Weight>(params[2]);
	auto W1 = std::make_shared<Weight>(params[3]);
	auto W2 = std::make_shared<Weight>(params[4]);

	auto B = std::make_shared<Bias>(params[5]);
	auto B3 = std::make_shared<Bias>(params[6]);
	auto B4 = std::make_shared<Bias>(params[7]);


	auto mul = std::make_shared<MUL>(X, W);
	auto sum = std::make_shared<SUM>(mul, B);
	auto relu = std::make_shared<ReLu>(sum);

//Dense Layer 1

	auto mul1 = std::make_shared<MUL>(relu, W1);

	auto sum3 = std::make_shared<SUM>(mul1, B3);
	auto relu3 = std::make_shared<ReLu>(sum3);

//Dense Layer 2

	auto mul2 = std::make_shared<MUL>(relu3, W2);
	auto sum4 = std::make_shared<SUM>(mul2, B4);

//    Output/Cost Layer
	auto soft = std::make_shared<Softmax>(sum4,10);
	auto CE = std::make_shared<CrossEntropyLoss>(soft,CN);

	//Create Deep Learning session
	Session session(CE, std::move(graph));


	if(train){

		session.run();

		std::cout << "LOSS:\n" << CE->getForward()(0,0) << std::endl;

//        std::cout << "\n Results of Last Run \n" << std::endl;
//        std::cout << "Output:\n" << soft->getForward() << std::endl;
//        std::cout << "LOSS:\n" << CE->getForward()(0,0) << std::endl;
//        std::cout<<"\nExpected:\n\n\n"<<CN->getForward()<<std::endl;
		params[2] = W->getForward();
		params[3] = W1->getForward();
		/*std::cout<<"\nparams:\n"<<params[4].block(0,0,3,3)<<std::endl;
		std::cout<<"\nForward:\n"<<W2->getForward().block(0,0,3,3)<<std::endl;*/

		params[4] = W2->getForward();

		params[5] = B->getForward();
		params[6] = B3->getForward();
		params[7] = B4->getForward();

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
			soft->getForward().block(i,0,1,10).maxCoeff(&maxRow,&maxCol);
			int p = maxCol;
			params[1].block(i,0,1,10).maxCoeff(&maxRow,&maxCol);
			int A = maxCol;
			if(p==A)correct++;
			total++;}
	}




}



int main() {
	int batch_size = 8;
	int epochs =30;
	int amount_batches =10;
	float correct=0,total=0;
	std::vector<Eigen::MatrixXf> training_data,training_label;
	std::vector<Eigen::MatrixXf> test_data,test_label;

	getBatches(batch_size,amount_batches,training_data,training_label);
	getBatches(batch_size,amount_batches,test_data,test_label,false);





	Eigen::MatrixXf W = generateRandomMatrix(0., 1, 784, 800);
	Eigen::MatrixXf W1 = generateRandomMatrix(0., 1, 800, 128);
	Eigen::MatrixXf W2 = generateRandomMatrix(0., 1,128, 10 );

	Eigen::MatrixXf b2 = Eigen::MatrixXf::Zero(batch_size,800);
	Eigen::MatrixXf b3 = Eigen::MatrixXf::Zero(batch_size,128);
	Eigen::MatrixXf b4 = Eigen::MatrixXf::Zero(batch_size, 10);
	std::vector<Eigen::MatrixXf> params ={training_data[0],training_label[0],W,W1,W2,b2,b3,b4};
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
		std::cout<<"Amount Correct: "<<correct<<"Amount TOTAL: "<<total<<"Percentage: "<<correct/(float)
				total<<std::endl;
		correct=0;
		total =0;
	}




}