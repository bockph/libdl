include_directories(include)
include_directories(Operations/include)
set(GRAPH_NAME graph)
set(GRAPH_SOURCES
        src/Node.cpp
        src/Operation.cpp
        src/Placeholder.cpp
        Operations/src/SummationOp.cpp
        Operations/src/MultiplicationOp.cpp
        Operations/src/SigmoidOP.cpp
        Operations/src/MSEOp.cpp
        src/Parameter.cpp
        Operations/src/MaxPoolOp.cpp
        Operations/src/ReLuOp.cpp
        Operations/src/SoftmaxOp.cpp
        Operations/src/CrossEntropyOp.cpp
        Operations/src/ConvolutionOp.cpp
        Operations/src/LossFunction.cpp
        Operations/src/ActivationFunction.cpp
        Operations/src/NormalFunction.cpp
        src/Graph.cpp
        Operations/src/OperationsFactory.cpp
        )
include_directories(${PROJECT_SOURCE_DIR}/extern/eigen/Eigen)

add_library(${GRAPH_NAME} ${GRAPH_SOURCES})
target_include_directories(${GRAPH_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/Operations/include)
target_link_libraries(${GRAPH_NAME} eigenI utils spdlogI catch2I)
set_target_properties(${GRAPH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_subdirectory(Operations/tests)
add_subdirectory(tests)
