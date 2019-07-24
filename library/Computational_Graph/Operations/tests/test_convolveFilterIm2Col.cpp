//
// Created by phili on 14.06.2019.
//

//
// Created by phili on 15.05.2019.
//
//
// Created by phili on 14.05.2019.
//
#include <MultiplicationOp.hpp>
#include <Session.hpp>
#include <Placeholder.hpp>
#include <memory>

#include <catch2/catch.hpp>
#include <iostream>

#include <ConvolveFilterIM2COL.hpp>


TEST_CASE("Im2Col ", "[method]") {

    Eigen::MatrixXf input(1,32);
    input<<1,2,1,1,
    2,1,2,2,
    1,1,1,1,
    2,2,2,2,1,1,1,1,
    2,2,2,2,
    1,2,1,2,
    2,1,2,1;

    Eigen::MatrixXf filter(2,8);
    filter<<1,0,0,1,0,1,1,0,
            2,0,0,2,
            0,2,2,0;


    Eigen::MatrixXf test(3,9);
    test<<1,2,3,3,3,3,5,5,5,
            2,1,3, 4,4,4,6,6,6,
            3,1,2,4,5,4,6,7,6;
    std::cout<<test<<std::endl;
//    test = test.reshaped<Eigen::RowMajor>(2,4);
        int batchSize=3;
        int size =3;
        int channel = 3;
        Matrix output(batchSize,size*channel);
    for(int i = 0;i<channel;i++){
        Matrix tmp = test.row(i);//block(0,i*size,channel,size);
        tmp=tmp.reshaped<Eigen::RowMajor>(batchSize,size).eval();
        output.block(0,i*size,batchSize,size)=tmp;
    }

    std::cout<<output<<std::endl;

    std::cout<<"hello\n"<<std::endl;

    Matrix output2(channel,batchSize*size);
    for(int i = 0;i<batchSize;i++){
        Matrix tmp = output.row(i);//block(0,i*size,channel,size);
        tmp=tmp.reshaped<Eigen::RowMajor>(channel,size).eval();
        output2.block(0,i*size,channel,size)=tmp;
    }
    std::cout<<output2<<std::endl;

    /*SECTION("Stride 1"){
        Eigen::MatrixXf result = ConvolveFilter::im2col(input,filter,1,2);

        Eigen::MatrixXf test(8,9);
        test<<  1, 2, 1, 2, 1, 2, 1, 1, 1,
                2, 1, 1, 1, 2, 2, 1, 1, 1,
                2, 1, 2, 1, 1, 1, 2, 2, 2,
                1, 2, 2, 1, 1, 1, 2, 2, 2,
                1, 1, 1, 2, 2, 2, 1, 2, 1,
                1, 1, 1, 2, 2, 2, 2, 1, 2,
                2, 2, 2, 1, 2, 1, 2, 1, 2,
                2, 2, 2, 2, 1, 2, 1, 2, 1;
        REQUIRE(result.isApprox(test));

        Eigen::MatrixXf result2 = ConvolveFilter::col2im(result,filter,1,2);

        REQUIRE(result2.isApprox(input));
    }
    *//*
     * Stride 2 has to work as well, as no information should be lost here
     *//*
    SECTION("Stride 2"){
        
        Eigen::MatrixXf result = ConvolveFilter::im2col(input,filter,2,2);
        std::cout<<result<<std::endl;


        Eigen::MatrixXf result2 = ConvolveFilter::col2im(result,filter,2,2);

        std::cout<<result2<<std::endl;
        REQUIRE(result2.isApprox(input));
    }*/



}


