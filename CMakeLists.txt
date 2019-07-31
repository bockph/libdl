cmake_minimum_required(VERSION 3.10)
project(libdl VERSION 0.1
        DESCRIPTION "testing CI setup"
        LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++17 -O0 -march=native")
#add external Libraries
add_subdirectory(extern)
#add Doxygen if installed (Code was taken from ELSA)
find_package(Doxygen)
if (DOXYGEN_FOUND)
    #set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen-config)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    #request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    add_custom_target(doc_doxygen
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
    endif(DOXYGEN_FOUND)

    add_subdirectory(extern/pybind11)
    add_subdirectory(library)

#copy sampe data to build dir
#This would be nice to use, so that the python applications could load the data from the build dir (where also the
#libdl is), but copying takes quite some time so it is commented and all paths set to project_dir/data
#file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/data)



    #<-------Stuff for directly building an Application----->
    set(TEST_TARGET_ONE Example_XOR)
    add_executable(${TEST_TARGET_ONE} Example_XOR.cpp)
    target_link_libraries(${TEST_TARGET_ONE} session eigenI)
    #target_compile_features(${TEST_TARGET_ONE} PUBLIC cxx_std_17)


    set(TEST_TARGET_TWO Example_MNIST)
    add_executable(${TEST_TARGET_TWO} Example_MNIST.cpp)
    target_link_libraries(${TEST_TARGET_TWO} session mnist utils)
    target_compile_definitions(${TEST_TARGET_TWO} PRIVATE MNIST_DATA_LOCATION="${CMAKE_SOURCE_DIR}/extern/mnist" WEIGHT_DEPOSIT="${CMAKE_SOURCE_DIR}/WeightDeposit/")

    #target_compile_features(${TEST_TARGET_TWO} PUBLIC cxx_std_17)


    set(TEST_TARGET_FOUR Example_LEGO)
    add_executable(${TEST_TARGET_FOUR} Example_LEGO.cpp)
    target_link_libraries(${TEST_TARGET_FOUR} session mnist utils lodepng)
    target_compile_definitions(${TEST_TARGET_FOUR} PRIVATE MNIST_DATA_LOCATION="${CMAKE_SOURCE_DIR}/extern/mnist" WEIGHT_DEPOSIT="${CMAKE_SOURCE_DIR}/WeightDeposit/" DATA_DIR="${CMAKE_SOURCE_DIR}/data/")

    #target_compile_features(${TEST_TARGET_FOUR} PUBLIC cxx_std_17)



