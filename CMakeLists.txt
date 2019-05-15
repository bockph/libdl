cmake_minimum_required(VERSION 3.10)
project(test VERSION 0.1
        DESCRIPTION "testing CI setup"
        LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 14)

add_subdirectory(extern)
add_subdirectory(library)
#<-------Stuff for directly building an Application----->
#set(TEST_TARGET testmain)
#add_executable(${TEST_TARGET} main.cpp )
#target_link_libraries (${TEST_TARGET} graph Catch2)#Eigen3::Eigen Catch2 spdlog)# libdl)# pybind11)
#target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)
#pybind11_add_module(pybindtest main.cpp)
#add_executable(testpy testPython.py )

