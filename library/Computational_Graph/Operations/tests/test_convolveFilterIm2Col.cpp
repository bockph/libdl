//
// Created by phili on 14.06.2019.
//

//
// Created by phili on 15.05.2019.
//
//
// Created by phili on 14.05.2019.
//
#include <MUL.hpp>
#include <Session.hpp>
#include <Placeholder.hpp>
#include <memory>
#include <Graph.hpp>

#include <catch2/catch.hpp>
#include <iostream>

#include <Filter.hpp>
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
    SECTION("Stride 1"){
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
    /*
     * Stride 2 has to work as well, as no information should be lost here
     */
    SECTION("Stride 2"){
        
        Eigen::MatrixXf result = ConvolveFilter::im2col(input,filter,2,2);
        std::cout<<result<<std::endl;


        Eigen::MatrixXf result2 = ConvolveFilter::col2im(result,filter,2,2);

        std::cout<<result2<<std::endl;
        REQUIRE(result2.isApprox(input));
    }



}


