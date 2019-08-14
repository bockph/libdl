add_subdirectory(LayerFactory)

set(LIB_NAME session)

set(LIB_SOURCES
        src/NeuralNetwork.cpp)
include_directories(${PROJECT_SOURCE_DIR}/extern/eigen/Eigen)

add_library(${LIB_NAME} ${LIB_SOURCES})
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(${LIB_NAME} graph utils eigenI layerFactory)
set_target_properties(${LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

