cmake_minimum_required(VERSION 3.10)
project(test VERSION 0.1
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

set(TEST_TARGET_TWO Example_MNIST)
add_executable(${TEST_TARGET_TWO} Example_MNIST.cpp)
#target_link_libraries (${TEST_TARGET_TWO} graph Eigen3::Eigen )# libdl)# pybind11)
target_link_libraries (${TEST_TARGET_TWO} graph  )# libdl)# pybind11)

target_compile_features(${TEST_TARGET_TWO} PUBLIC cxx_std_17)

#pybind11_add_module(pybindtest main.cpp)
#add_executable(testpy testPython.py )

