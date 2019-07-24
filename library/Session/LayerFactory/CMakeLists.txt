


include_directories(include)
set(LIB_NAME layerInterface)

set(LIB_SOURCES
        src/AbstractLayer.cpp
        src/ConvolutionLayer.cpp
        src/DenseLayer.cpp
        src/InputLayer.cpp
        src/LogitsLayer.cpp
        src/LossLayer.cpp
        src/MaxPoolLayer.cpp
 )
include_directories(${PROJECT_SOURCE_DIR}/extern/eigen/Eigen)

add_library(${LIB_NAME}  ${LIB_SOURCES})
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(${LIB_NAME} graph utils eigenI spdlogI catch2I pybind11I)
set_target_properties(${LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

