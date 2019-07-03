cmake_minimum_required(VERSION 3.10)
project(libdl VERSION 0.1
        DESCRIPTION "testing CI setup"
        LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 14)
#find_package(Eigen3 REQUIRED)
add_subdirectory(extern)
add_subdirectory(library)
#<-------Stuff for directly building an Application----->
set(TEST_TARGET_ONE Example_XOR)
add_executable(${TEST_TARGET_ONE} Example_XOR.cpp)
#target_link_libraries (${TEST_TARGET_ONE} graph Eigen3::Eigen )# libdl)# pybind11)
target_link_libraries (${TEST_TARGET_ONE} graph eigenI spdlogI catch2I pybind11I )# libdl)# pybind11)
target_compile_features(${TEST_TARGET_ONE} PUBLIC cxx_std_17)

set(TEST_TARGET_ONE_2 Example_XOR_2)
add_executable(${TEST_TARGET_ONE_2} Example_XOR_2.cpp)
#target_link_libraries (${TEST_TARGET_ONE} graph Eigen3::Eigen )# libdl)# pybind11)
target_link_libraries (${TEST_TARGET_ONE_2} graph eigenI spdlogI catch2I pybind11I )# libdl)# pybind11)
target_compile_features(${TEST_TARGET_ONE_2} PUBLIC cxx_std_17)

set(TEST_TARGET_TWO Example_MNIST)
add_executable(${TEST_TARGET_TWO} Example_MNIST.cpp)
#target_link_libraries (${TEST_TARGET_TWO} graph Eigen3::Eigen )# libdl)# pybind11)
target_link_libraries (${TEST_TARGET_TWO} graph mnist )# libdl)# pybind11)
target_compile_definitions(${TEST_TARGET_TWO} PRIVATE MNIST_DATA_LOCATION="${CMAKE_SOURCE_DIR}/extern/mnist")

target_compile_features(${TEST_TARGET_TWO} PUBLIC cxx_std_17)

set(TEST_TARGET_TWO_2 Example_MNIST_2)
add_executable(${TEST_TARGET_TWO_2} Example_MNIST_2.cpp)
#target_link_libraries (${TEST_TARGET_TWO} graph Eigen3::Eigen )# libdl)# pybind11)
target_link_libraries (${TEST_TARGET_TWO_2} graph mnist )# libdl)# pybind11)
target_compile_definitions(${TEST_TARGET_TWO_2} PRIVATE MNIST_DATA_LOCATION="${CMAKE_SOURCE_DIR}/extern/mnist")

target_compile_features(${TEST_TARGET_TWO_2} PUBLIC cxx_std_17)

set(TEST_TARGET_TWO_3 Example_MNIST_3)
add_executable(${TEST_TARGET_TWO_3} Example_MNIST3.cpp)
#target_link_libraries (${TEST_TARGET_TWO} graph Eigen3::Eigen )# libdl)# pybind11)
target_link_libraries (${TEST_TARGET_TWO_3} graph mnist )# libdl)# pybind11)
target_compile_definitions(${TEST_TARGET_TWO_3} PRIVATE MNIST_DATA_LOCATION="${CMAKE_SOURCE_DIR}/extern/mnist")

target_compile_features(${TEST_TARGET_TWO_3} PUBLIC cxx_std_17)

set(TEST_TARGET_TWO_4 Example_MNIST_4)
add_executable(${TEST_TARGET_TWO_4} Example_MNIST_4.cpp)
#target_link_libraries (${TEST_TARGET_TWO_4} graph Eigen3::Eigen )# libdl)# pybind11)
target_link_libraries (${TEST_TARGET_TWO_4} graph mnist )# libdl)# pybind11)
target_compile_definitions(${TEST_TARGET_TWO_4} PRIVATE MNIST_DATA_LOCATION="${CMAKE_SOURCE_DIR}/extern/mnist")

target_compile_features(${TEST_TARGET_TWO_4} PUBLIC cxx_std_17)


#pybind11_add_module(pybindtest main.cpp)
#add_executable(testpy testPython.py )

