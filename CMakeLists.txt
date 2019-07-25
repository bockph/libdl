cmake_minimum_required(VERSION 3.10)
project(libdl VERSION 0.1
        DESCRIPTION "testing CI setup"
        LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/include/python2.7")#-O0 -march=native")
#find_package(Eigen3 REQUIRED)
add_subdirectory(extern)
add_subdirectory(extern/pybind11)
add_subdirectory(library)
add_subdirectory(pythonApps)
#<-------Stuff for directly building an Application----->
set(TEST_TARGET_ONE Example_XOR)
add_executable(${TEST_TARGET_ONE} Example_XOR.cpp)
target_link_libraries (${TEST_TARGET_ONE} graph eigenI spdlogI catch2I pybind11I )# libdl)# pybind11)
target_compile_features(${TEST_TARGET_ONE} PUBLIC cxx_std_17)



set(TEST_TARGET_TWO Example_MNIST)
add_executable(${TEST_TARGET_TWO} Example_MNIST.cpp)
target_link_libraries (${TEST_TARGET_TWO} graph mnist utils )
target_compile_definitions(${TEST_TARGET_TWO} PRIVATE MNIST_DATA_LOCATION="${CMAKE_SOURCE_DIR}/extern/mnist" WEIGHT_DEPOSIT="${CMAKE_SOURCE_DIR}/WeightDeposit/")

target_compile_features(${TEST_TARGET_TWO} PUBLIC cxx_std_17)

set(TEST_TARGET_THREE Example_MNIST_Layer)
add_executable(${TEST_TARGET_THREE} Example_MNIST_Layer.cpp)
target_link_libraries (${TEST_TARGET_THREE} session mnist utils )
target_compile_definitions(${TEST_TARGET_THREE} PRIVATE MNIST_DATA_LOCATION="${CMAKE_SOURCE_DIR}/extern/mnist" WEIGHT_DEPOSIT="${CMAKE_SOURCE_DIR}/WeightDeposit/")

target_compile_features(${TEST_TARGET_THREE} PUBLIC cxx_std_17)

set(TEST_TARGET_FOUR Example_LEGO)
add_executable(${TEST_TARGET_FOUR} Example_LEGO.cpp)
target_link_libraries (${TEST_TARGET_FOUR} session mnist utils lodepng )
target_compile_definitions(${TEST_TARGET_FOUR} PRIVATE MNIST_DATA_LOCATION="${CMAKE_SOURCE_DIR}/extern/mnist" WEIGHT_DEPOSIT="${CMAKE_SOURCE_DIR}/WeightDeposit/" DATA_DIR="${CMAKE_SOURCE_DIR}/data/")

target_compile_features(${TEST_TARGET_FOUR} PUBLIC cxx_std_17)



